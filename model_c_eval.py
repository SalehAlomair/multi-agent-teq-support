"""Comparable completion-only loss and auditable generation for Model C."""
import math
from contextlib import nullcontext
import torch


def encode_conversation(tokenizer, messages, max_length=512):
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
    ids = tokenizer(full, add_special_tokens=False)['input_ids']
    if ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError('Chat template/tokenization does not preserve the prompt prefix.')
    if len(ids) > max_length or len(ids) <= len(prompt_ids):
        raise ValueError('Conversation exceeds limit or has no completion tokens.')
    return {'input_ids': ids, 'attention_mask': [1] * len(ids),
            'labels': [-100] * len(prompt_ids) + ids[len(prompt_ids):]}


def completion_metrics(model, tokenizer, rows, baseline=False):
    was_training = model.training
    total, count = 0.0, 0
    model.eval()
    try:
        with (model.disable_adapter() if baseline else nullcontext()), torch.no_grad():
            for row in rows:
                encoded = encode_conversation(tokenizer, row['messages'])
                inputs = {k: torch.tensor([v], device=model.get_input_embeddings().weight.device)
                          for k, v in encoded.items()}
                tokens = int((inputs['labels'][:, 1:] != -100).sum())
                loss = float(model(**inputs, use_cache=False).loss)
                if not math.isfinite(loss):
                    raise ValueError('Non-finite loss.')
                total += loss * tokens
                count += tokens
    finally:
        model.train(was_training)
    if not count:
        raise ValueError('Empty evaluation dataset.')
    loss = total / count
    return {'loss': loss, 'perplexity': math.exp(loss) if loss < 700 else None,
            'target_tokens': count}


def generate_answer(model, tokenizer, messages, max_new_tokens=384, baseline=False):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False)
    inputs = inputs.to(model.get_input_embeddings().weight.device)
    was_training = model.training
    model.eval()
    try:
        with (model.disable_adapter() if baseline else nullcontext()), torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                    use_cache=True, pad_token_id=tokenizer.pad_token_id,
                                    eos_token_id=tokenizer.eos_token_id)
        ids = output[0, inputs['input_ids'].shape[1]:].tolist()
        ended = bool(ids and ids[-1] == tokenizer.eos_token_id)
        return {'text': tokenizer.decode(ids, skip_special_tokens=True),
                'generated_tokens': len(ids), 'finish_reason': 'eos' if ended else 'length',
                'truncated': not ended and len(ids) >= max_new_tokens}
    finally:
        model.train(was_training)


def adapter_diagnostics(model, tokenizer, messages):
    """Measure adapter effect on identical inputs without changing weights."""
    if not hasattr(model, 'disable_adapter'):
        raise ValueError('Expected an unmerged PEFT model.')
    from peft.tuners.tuners_utils import BaseTunerLayer
    layers = [m for m in model.modules() if isinstance(m, BaseTunerLayer)]
    if not layers or any(layer.merged or layer.disable_adapters for layer in layers):
        raise ValueError('Adapter layers must be enabled and unmerged.')
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False)
    inputs = inputs.to(model.get_input_embeddings().weight.device)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            with model.disable_adapter():
                base = model(**inputs, use_cache=False).logits[:, -1, :].float().cpu()
            tuned = model(**inputs, use_cache=False).logits[:, -1, :].float().cpu()
        delta = (tuned - base).abs()
        weights = [p.detach() for name, p in model.named_parameters() if 'lora_B' in name]
        return {
            'active_adapters': list(model.active_adapters),
            'adapter_layers': len(layers),
            'lora_b_nonzero_elements': sum(int(torch.count_nonzero(p)) for p in weights),
            'max_logit_difference': float(delta.max()),
            'mean_logit_difference': float(delta.mean()),
            'effect_detected': bool(delta.max() > 0),
            'note': 'Nonzero B weights are evidence of updates for this zero-B initialization. '
                    'Logit differences establish an effect on this probe, not better behavior.',
        }
    finally:
        model.train(was_training)

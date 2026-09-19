import unittest
from model_c_eval import encode_conversation

class Tokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return ''.join(m['content'] for m in messages)
    def __call__(self, text, **kwargs):
        return {'input_ids': [ord(c) for c in text]}

class EncodingTests(unittest.TestCase):
    def test_masks_only_prompt(self):
        result = encode_conversation(Tokenizer(), [{'content':'question'}, {'content':'answer'}])
        self.assertEqual(result['labels'][:8], [-100]*8)
        self.assertEqual(result['labels'][8:], list(map(ord, 'answer')))
    def test_rejects_truncation_and_empty_completion(self):
        for answer, limit in [('long answer', 3), ('', 512)]:
            with self.assertRaises(ValueError):
                encode_conversation(Tokenizer(), [{'content':'q'}, {'content':answer}], limit)

if __name__ == '__main__':
    unittest.main()

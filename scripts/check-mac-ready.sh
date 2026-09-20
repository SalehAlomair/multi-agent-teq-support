#!/usr/bin/env bash

set -eu

cd "$(dirname "$0")/.."

fail() {
    echo "NOT READY: $1" >&2
    exit 1
}

for command_name in docker curl git; do
    command -v "$command_name" >/dev/null 2>&1 || fail "$command_name is not installed"
done

git lfs version >/dev/null 2>&1 || fail "Git LFS is not installed. Run: brew install git-lfs && git lfs install"

test -f .env || fail ".env is missing. Run: cp .env.example .env, then add the real local values"

get_env_value() {
    sed -n "s/^$1=//p" .env | tail -1
}

openai_key=$(get_env_value OPENAI_API_KEY)
webui_secret=$(get_env_value WEBUI_SECRET_KEY)

case "$openai_key" in
    ""|replace_*) fail "OPENAI_API_KEY still needs a real value in .env" ;;
esac

case "$webui_secret" in
    ""|replace_*) fail "WEBUI_SECRET_KEY still needs a real value in .env" ;;
esac

check_model() {
    model_path="$1"
    minimum_size="$2"

    test -f "$model_path" || fail "$model_path is missing. Run: git lfs pull"
    actual_size=$(wc -c < "$model_path")
    test "$actual_size" -ge "$minimum_size" || fail "$model_path is too small and may still be a Git LFS pointer"
}

check_model models/model_a/intent_classifier/model.safetensors 200000000
check_model models/model_b/qa_model/model.safetensors 200000000
check_model models/model_c/support_adapter_v2/adapter_model.safetensors 1000000

docker compose config --quiet || fail "docker-compose.yml or .env is invalid"

llama_url=$(get_env_value LLAMA_SERVER_DOCKER_URL)
test -n "$llama_url" || fail "LLAMA_SERVER_DOCKER_URL is missing from .env"

llama_base=${llama_url%/chat/completions}
curl --fail --silent --show-error --max-time 10 "$llama_base/models" >/dev/null \
    || fail "The Mac cannot reach llama-server at $llama_base/models"

echo "READY: model artifacts, Docker configuration, and the remote Qwen server are available."
echo "Start with: docker compose up --build -d"

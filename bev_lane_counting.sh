#!/bin/bash

set -e

HTTPS_PROXY="http://localhost"
API_KEY="****"
OPENROUTER_API_KEY="****" # OpenRouter
# Azure OpenAI
GPT5MINI="****"
GPT5NANO="****"
AZURE_API_KEY="****"

SCRIPT="bev_lane_counting.py"

# 重复执行3次以修正意外错误
for i in {1..3}; do
    # uv run $SCRIPT -b https://cloud.infini-ai.com/maas/v1 -m qwen2.5-vl-72b-instruct -c 20 -k $API_KEY
    # uv run $SCRIPT -b https://cloud.infini-ai.com/maas/v1 -m qwen2.5-vl-32b-instruct -c 20 -k $API_KEY
    # uv run $SCRIPT -b https://cloud.infini-ai.com/maas/v1 -m qwen2.5-vl-7b-instruct -c 20 -k $API_KEY
    # uv run $SCRIPT -b https://cloud.infini-ai.com/maas/v1 -m glm-4.5v -c 20 -k $API_KEY

    # uv run $SCRIPT -b https://openrouter.ai/api/v1 -m google/gemini-2.5-pro -c 20 -k $OPENROUTER_API_KEY
    # uv run $SCRIPT -b https://openrouter.ai/api/v1 -m google/gemini-2.5-flash -c 20 -k $OPENROUTER_API_KEY
    # uv run $SCRIPT -b https://openrouter.ai/api/v1 -m google/gemini-2.5-flash-image-preview -c 20 -k $OPENROUTER_API_KEY
    # uv run $SCRIPT -b https://openrouter.ai/api/v1 -m google/gemma-3-27b-it -c 20 -k $OPENROUTER_API_KEY
    # uv run $SCRIPT -b https://openrouter.ai/api/v1 -m google/gemma-3-12b-it -c 20 -k $OPENROUTER_API_KEY
    # uv run $SCRIPT -b https://openrouter.ai/api/v1 -m meta-llama/llama-3.2-90b-vision-instruct -c 20 -k $OPENROUTER_API_KEY
    # uv run $SCRIPT -b https://openrouter.ai/api/v1 -m meta-llama/llama-3.2-11b-vision-instruct -c 20 -k $OPENROUTER_API_KEY
    uv run $SCRIPT -b https://openrouter.ai/api/v1 -m openai/gpt-5 -c 20 -k $OPENROUTER_API_KEY
    uv run $SCRIPT -b $GPT5MINI -m gpt-5-mini -c 20 -k $AZURE_API_KEY
    uv run $SCRIPT -b $GPT5NANO -m gpt-5-nano -c 20 -k $AZURE_API_KEY
done

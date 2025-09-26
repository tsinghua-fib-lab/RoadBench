#!/bin/bash

set -e

HTTPS_PROXY="http://localhost"
API_KEY="****"
OPENROUTER_API_KEY="****" # OpenRouter
# Azure OpenAI
GPT5MINI="****"
GPT5NANO="****"
AZURE_API_KEY="****"

SCRIPT="bev_lane_designations.py"

# 列出命令后缀组合
# 定义额外后缀组合列表，每个元素为字符串，表示要追加到命令行的参数
EXTRA_SUFFIXES=(
    "--reference_line_mode prompt_only --direction_annotation_mode colors"
    "--reference_line_mode image_only --direction_annotation_mode arrows"
    "--reference_line_mode image_only --direction_annotation_mode colors"
    "--reference_line_mode both --direction_annotation_mode colors"
)

# 在循环外遍历这些额外后缀，执行命令
for SUFFIX in "${EXTRA_SUFFIXES[@]}"; do
for i in {1..3}; do
    uv run $SCRIPT -b $GPT5MINI -m gpt-5-mini -c 20 -k $AZURE_API_KEY $SUFFIX
    uv run $SCRIPT -b https://cloud.infini-ai.com/maas/v1 -m glm-4.5v -c 20 -k $API_KEY $SUFFIX
done
done

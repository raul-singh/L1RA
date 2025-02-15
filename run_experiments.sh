#!/bin/bash
set -xe

# Baselines
conda run -n l1ra_baseline python ./src/script/train_assistant.py --config-path ./resources/configs/baselines/mistral.yml
conda run -n l1ra_baseline python ./src/script/train_assistant.py --config-path ./resources/configs/baselines/llama.yml
conda run -n l1ra_baseline python ./src/script/train_assistant.py --config-path ./resources/configs/baselines/mistral_alt.yml
conda run -n l1ra_baseline python ./src/script/train_assistant.py --config-path ./resources/configs/baselines/llama_alt.yml
# Experiments
conda run -n l1ra_baseline python ./src/script/train_assistant.py --config-path ./resources/configs/mistral.yml
conda run -n l1ra_baseline python ./src/script/train_assistant.py --config-path ./resources/configs/llama.yml

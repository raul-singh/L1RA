#!/bin/bash
set -xe

# Baselines
conda run -n l1ra_baseline python ./src/script/train_assistant.py --config-path ./resources/configs/baselines/mistral.yml
conda run -n l1ra_baseline python ./src/script/train_assistant.py --config-path ./resources/configs/baselines/llama3.yml
# conda run -n l1ra_baseline python ./src/script/train_assistant.py --config-path ./resources/configs/baselines/mistral_alt.yml
# conda run -n l1ra_baseline python ./src/script/train_assistant.py --config-path ./resources/configs/baselines/llama3_alt.yml
# Experiments
conda run -n l1ra python ./src/script/train_assistant.py --config-path ./resources/configs/mistral.yml
conda run -n l1ra python ./src/script/train_assistant.py --config-path ./resources/configs/llama3.yml

import logging
import os
import time
from datetime import datetime

import click
import numpy as np
import pandas as pd
import peft
import torch
import yaml
from datasets import DatasetDict, load_dataset
from peft import AdaLoraConfig, LoraConfig
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from trl import SFTTrainer

from l1ra import L1RAConfig, L1RAModel, L1RASFTTrainer, L1RATrainer

ADAPTER_CONFIG_TO_TRAINER_MAPPING = {
    L1RAConfig: L1RASFTTrainer,
    LoraConfig: SFTTrainer,
    AdaLoraConfig: SFTTrainer, #TODO
}


chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif false == true and not '<<SYS>>' in messages[0]['content'] %}{% set loop_messages = messages %}{% set system_message = 'You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\\n\\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\\'t know the answer to a question, please don\\'t share false information.' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'system' %}{{ '<<SYS>>\\n' + content.strip() + '\\n<</SYS>>\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"


# Create and initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    encoding="utf-8",
    format="%(name)s %(levelname)s: %(message)s",
    level=logging.INFO
)


def load_config(path: str):
    with open(path, "r") as file:
        config = yaml.safe_load(file)

    return config


def load_guanaco(config, tokenizer, validation_split=0.0):
    seed = config.get("seed", 42)
    dataset_id = config["dataset_id"]

    dataset = load_dataset(dataset_id)

    test_ds = dataset["test"]

    if validation_split > 0.0:
        train_val = dataset["train"].train_test_split(validation_split, seed=seed)
        train_ds = train_val["train"]
        val_ds = train_val["test"]

    dataset = DatasetDict(
        {"train": train_ds, "validation": val_ds, "test": test_ds}
    )

    def preprocess(example):
        s = example["text"].split("### ")
        chat = []
        for msg in s:
            if msg.startswith("Human: "):
                chat.append(
                    {"role": "user", "content": msg[len("Human: "):]}
                )
            elif msg.startswith("Assistant: "):
                chat.append(
                    {"role": "assistant", "content": msg[len("Assistant: "):]}
                )
        chat = tokenizer.apply_chat_template(chat, tokenize=False)
        return {"text": chat}

    return dataset.map(preprocess)


def load_and_preprocess_dataset(config, tokenizer):
    if tokenizer.chat is None:
        tokenizer.chat_template = chat_template

    dataset_id = config["dataset_id"]

    if dataset_id == "timdettmers/openassistant-guanaco":
        return load_guanaco(config, tokenizer)

    else:
        raise NotImplementedError(f"There is no implemented pipeline for {dataset_id}.")


def tokenize_dataset(dataset, tokenizer):
    def tokenize_function(examples):
        input_encodings = tokenizer(examples["text"], return_tensors="pt", padding=True, truncation=True)
        sample = {"input_ids": input_encodings.input_ids.cuda()}
        return sample

    return dataset.map(tokenize_function, batched=True)


def create_model(config):
    q_bit = config["quantization_bit"]
    model_id = config["model_id"]

    if q_bit == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config,
        trus_remote_code=True,
    )
    model.config.use_cache = False
    return model


def create_adapter_config(config, adapter_type):
    adapter_kwargs = config["adapter_config"]

    if adapter_type == "l1ra":
        config_cls = L1RAConfig
        adapter_kwargs.update(config.get("l1ra_specific_args", {}))
    elif adapter_type == "lora":
        config_cls = LoraConfig
    elif adapter_type == "adalora":
        config_cls ==AdaLoraConfig
        adapter_kwargs.update(config.get("adalora_specific_args", {}))

    return config_cls(**adapter_kwargs)


def compute_n_adapter_params(model):
    params = 0

    for n,p in model.named_parameters():
        if "lora" in n:
            params += p.numel()

    return params


def rank_evolution(trainer, model_id):
    model = trainer.model.base_model
    rank_evolution = model.rank_evolution

    n_layers = AutoConfig.from_pretrained(model_id).num_hidden_layers
    model_shape = (n_layers, len(rank_evolution[0])//n_layers)
    training_steps = trainer.num_training_steps * trainer.args.gradient_accumulation_steps
    update_steps = training_steps * model.peft_config["default"].rank_update_ratio

    ranks = []
    for rank in rank_evolution:
        rank = np.array(rank)
        rank = rank.reshape(model_shape)
        ranks.append(rank)

    tuples = []
    for i, distr in enumerate(ranks, 1):
        step = update_steps*i
        for l, layer in enumerate(distr):
            layer = layer.tolist()
            layer.append(step)
            layer.append(l)
            layer = tuple(layer)
            tuples.append(layer)


    df = pd.DataFrame(
        tuples,
        columns=[
            "$W_{q}$",
            "$W_{k}$",
            "$W_{v}$",
            "$W_o$",
            "$W_{gate}$",
            "$W_{up}$",
            "$W_{down}$",
            "step",
            "layer"
            ]
        )

    return df


def train_and_evaluate(model, tokenizer, adapter_config, dataset, config):
    training_args = TrainingArguments(**config["training_args"])
    trainer_cls = ADAPTER_CONFIG_TO_TRAINER_MAPPING[type(adapter_config)]

    start = time.time()

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation", None),
        dataset_text_field="text",
        peft_config=adapter_config,
        max_seq_length=config["max_seq_length"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        tokenizer=tokenizer
    )

    trainer.train()

    end = time.time()

    history = pd.DataFrame(trainer.state.log_history)
    adapter_params = compute_n_adapter_params(model)
    peak_mem_usage = torch.cuda.max_memory_allocated()
    time_taken = end - start

    tokenized_dataset = tokenize_dataset(dataset)
    test_loss = trainer.evaluate(eval_dataset=tokenized_dataset["test"])["eval_loss"]
    ppl = float(np.exp(test_loss))

    report = {
        "history": history,
        "adapter_params": adapter_params,
        "peak_mem_usage": peak_mem_usage,
        "time_taken": time_taken,
        "ppl": ppl,
    }

    if isinstance(adapter_config, L1RAConfig):
        report["rank_evolution"] = rank_evolution(trainer, config["model_id"])

    return report


def save_report(adapter_type, report):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    directory = os.path.join("experiments", f"{adapter_type}-{timestamp}")
    os.makedirs(directory)

    report["history"].to_csv(os.path.join(directory, "history.csv"))
    report.pop("history")

    if "rank_evolution" in report:
        report["rank_evolution"].to_csv(os.path.join(directory, "rank_evolution.csv"))
        report.pop("rank_evolution")

    with open(os.path.join(directory, "report.yml"), "w") as file:
        yaml.dump(report, file)


@click.command()
@click.option('--config-path', help='Path of training config file.')
@click.option('--cv', is_flag=True, help='Perform cross-validation.')
def main(config_path, cv):
    config = load_config(config_path)

    model_id = config["model_id"]
    max_seq_len = config["max_seq_length"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        model_max_length=max_seq_len,
        padding_side="right",
    )
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(config)

    for adapter_type in config["to_train"]:
        adapter_config = create_adapter_config(config, adapter_type)
        model = create_model(config)
        report = train_and_evaluate(model, tokenizer, adapter_config, dataset, config)
        save_report(adapter_type, report)


if __name__ == '__main__':
    main()
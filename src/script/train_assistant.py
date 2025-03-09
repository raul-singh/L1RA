import logging
import os
import time
import gc
from datetime import datetime
from itertools import product
from copy import deepcopy
from shutil import copy2
import click
import math
import numpy as np
import pandas as pd
import torch
import yaml
from datasets import DatasetDict, load_dataset
from peft import AdaLoraConfig, LoraConfig, prepare_model_for_kbit_training, get_peft_model
from sklearn.model_selection import KFold
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

from l1ra import L1RAConfig, L1RASFTTrainer

ADAPTER_CONFIG_TO_TRAINER_MAPPING = {
    L1RAConfig: L1RASFTTrainer,
    LoraConfig: SFTTrainer,
    AdaLoraConfig: SFTTrainer,  # TODO
}

chat_template = '''{% if (messages | first).role != 'system' %}The following is a chat between a human user (referred to as "User") and an AI assistant (referred to as "Assistant") knowledgeable in all sort of subjects. 
The assistant is very respectful, honest and it always answer as helpfully as possible, while being safe. 
The answers of the assistant never include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
When questions do not make any sense or are not factually correct, the assistant avoid answering incorrect information and explain why it has not understood and asks for a clarification.

---

{% endif %}{% for message in messages %}{% if message.role == 'system' and loop.index0 == 0 %}{{ message.content | trim }}

---

{% elif message.role == 'user' %}>>> User: {{ message.content | trim }}

{% elif message.role == 'assistant' %}>>> Assistant: {{ message.content | trim }}

{% endif %}{% endfor %}'''

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


def load_guanaco(config, tokenizer, validation_split=0.1):
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


def load_open_orca(config, tokenizer, validation_split=0.1, test_split=0.1):
    seed = config.get("seed", 42)
    dataset_id = config["dataset_id"]

    dataset = load_dataset(dataset_id, split='train')

    if test_split > 0.0 and validation_split > 0.0:
        train_test = dataset.train_test_split(validation_split, seed=seed)
        dataset = train_test["train"]
        test_ds = train_test["test"]

        train_val = dataset.train_test_split(validation_split, seed=seed)
        train_ds = train_val["train"]
        val_ds = train_val["test"]

        dataset = DatasetDict(
            {"train": train_ds, "validation": val_ds, "test": test_ds}
        )
    elif test_split > 0.0:
        train_test = dataset.train_test_split(validation_split, seed=seed)
        train_ds = train_test["train"]
        test_ds = train_test["test"]

        dataset = DatasetDict(
            {"train": train_ds, "test": test_ds}
        )
    elif validation_split > 0.0:
        train_val = dataset.train_test_split(validation_split, seed=seed)
        train_ds = train_val["train"]
        val_ds = train_val["test"]

        dataset = DatasetDict(
            {"train": train_ds, "validation": val_ds}
        )

    def preprocess(example):
        chat = []
        if example['system_prompt']:
            chat.append(
                {"role": "system", "content": example['system_prompt']}
            )
        chat.append(
            {"role": "user", "content": example['question']}
        )
        chat.append(
            {"role": "assistant", "content": example['response']}
        )
        chat = tokenizer.apply_chat_template(chat, tokenize=False)
        return {"text": chat + tokenizer.eos_token}

    return dataset.map(preprocess)


def load_and_preprocess_dataset(config, tokenizer):
    if tokenizer.chat_template is None:
        tokenizer.chat_template = chat_template

    dataset_id = config["dataset_id"]

    if dataset_id == "timdettmers/openassistant-guanaco":
        dataset = load_guanaco(config, tokenizer)
    elif dataset_id == 'Open-Orca/OpenOrca':
        dataset = load_open_orca(config, tokenizer)
    else:
        raise NotImplementedError(
            f"There is no implemented pipeline for {dataset_id}."
        )

    logger.info("%s dataset loaded and preprocessed.", dataset_id)

    if "subset" in config:
        subset = config["subset"]
        for d in dataset.keys():
            dataset[d] = dataset[d].shuffle().select(
                list(range(int(dataset[d].num_rows * subset)))
            )
        logger.info(f"Created {subset * 100} % subset of dataset.")

    return dataset


def tokenize_dataset(dataset, tokenizer):
    def tokenize_function(examples):
        input_encodings = tokenizer(
            examples["text"],
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        sample = {"input_ids": input_encodings.input_ids.to('cuda:0')}
        return sample

    return dataset.map(tokenize_function, batched=True)


def create_model(config, adapter_config):
    q_bit = config["quantization_bit"]
    model_id = config["model_id"]
    token=config.get('token')

    if q_bit == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        logger.info("Quantizing model to %d-bit", q_bit)
    else:
        bnb_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # device_map='cuda:0',
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        # low_cpu_mem_usage=False,
        token=token
    ).to('cuda:0')
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    logger.info("%s loaded.", model_id)

    model = get_peft_model(model, adapter_config)

    return model


def create_adapter_config(config, adapter_type, dataset=None):
    adapter_kwargs = config["adapter_config"]
    # NOTE: When using gradient accumulation, one step is counted as one step with backward pass (see https://huggingface.co/docs/transformers/en/main_classes/trainer)

    if adapter_type == "l1ra":
        config_cls = L1RAConfig
        adapter_kwargs.update(config.get("l1ra_specific_args", {}))
    elif adapter_type == "lora":
        config_cls = LoraConfig
    elif adapter_type == "adalora":
        training_steps = int(math.ceil(
            len(dataset["train"]) / (
                config['training_args'].get('per_device_train_batch_size', 1) *
                config['training_args'].get('gradient_accumulation_steps', 1)
            ))
        )
        warmup_steps = max(
            config['training_args'].get('warmup_steps', 0),
            int(math.ceil(training_steps * config['training_args'].get('warmup_ratio', 0.0)))
        )
        config_cls = AdaLoraConfig
        adapter_kwargs.update(
            config.get("adalora_specific_args", {}) | {'total_step': training_steps, 'tinit': warmup_steps}
        )
        if 'budget_update_ratio' in adapter_kwargs:
            # DeltaT shouldn't always be 1 (see this example: https://github.com/huggingface/peft/blob/main/examples/int8_training/peft_adalora_whisper_large_training.py).
            # We will set in configs same frequency of L1RA rank updates for fair comparison
            adapter_kwargs['deltaT'] = max(1, int(math.ceil(training_steps * adapter_kwargs.pop('budget_update_ratio'))))
    else:
        raise ValueError()

    logger.info("Loading adapter config: %s", adapter_kwargs)
    return config_cls(**adapter_kwargs)


def compute_n_adapter_params(model):
    params = 0

    for n, p in model.named_parameters():
        if "lora" in n:
            params += p.numel()

    return params


def rank_evolution(trainer, model_id):
    model = trainer.model.base_model
    rank_evolution = model.rank_evolution

    n_layers = AutoConfig.from_pretrained(model_id).num_hidden_layers
    model_shape = (n_layers, len(rank_evolution[0])//n_layers)
    training_steps = trainer.num_training_steps * trainer.args.gradient_accumulation_steps
    update_steps = int(training_steps * model.peft_config["default"].rank_update_ratio)

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
    args = config["training_args"]
    #args.update({"dataset_text_field": "text", "max_seq_length": config["max_seq_length"]})
    training_args = TrainingArguments(**args)
    trainer_cls = ADAPTER_CONFIG_TO_TRAINER_MAPPING[type(adapter_config)]

    logger.debug("Training %s with args:\n%s", config["model_id"], training_args)

    start = time.time()

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation", None),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=config["max_seq_length"]
    )

    trainer.train()

    end = time.time()

    history = pd.DataFrame(trainer.state.log_history)
    adapter_params = compute_n_adapter_params(model)
    peak_mem_usage = torch.cuda.max_memory_allocated()
    time_taken = end - start

    logger.info("Model successfully trained.")

    logger.info("%s", [p for n,p in model.named_parameters() if "lora_c" in n])

    trainer.model.eval()
    if isinstance(adapter_config, L1RAConfig):
        logger.info("l1ra lambda: %f", trainer.model.peft_config["default"].l1ra_lambda)

    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    test_loss = trainer.evaluate(eval_dataset=tokenized_dataset["test"])["eval_loss"]
    ppl = float(np.exp(test_loss))
    logger.info("Test loss: %f", test_loss)
    logger.info("Test perplexity: %f", ppl)

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


def save_report(adapter_type, report, config_file_path):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    directory = os.path.join("experiments", f"{adapter_type}-{timestamp}")
    os.makedirs(directory)

    copy2(config_file_path, directory)

    report["history"].to_csv(
        os.path.join(directory, "history.csv"), index=False
    )
    report.pop("history")

    if "rank_evolution" in report:
        report["rank_evolution"].to_csv(
            os.path.join(directory, "rank_evolution.csv"),
            index=False
        )
        report.pop("rank_evolution")

    with open(os.path.join(directory, "report.yml"), "w") as file:
        yaml.dump(report, file)

    logger.info("Training report saved in %s", directory)


def load_tokenizer(config):
    model_id = config["model_id"]
    max_seq_len = config["max_seq_length"]
    token=config.get('token')

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        model_max_length=max_seq_len,
        padding_side="right",
        token=token
    )
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("%s tokenizer loaded.", model_id)

    return tokenizer


def cross_validation(cv_config, run_config):
    K = cv_config["cv_k"]
    kf = KFold(n_splits=K)

    tokenizer = load_tokenizer(run_config)
    dataset = load_and_preprocess_dataset(run_config, tokenizer)
    cv_report = []

    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    directory = os.path.join("experiments", f"cv-{run_config['model_id']}-{timestamp}")
    os.makedirs(directory)

    cv_configs = (
        dict(zip(cv_config['param_grid'].keys(), values)) for values in product(*cv_config['param_grid'].values())
    )

    for cv_config_ in cv_configs:

        fold_reports = []

        logger.info(f"Performing {K}-fold cross-validation -- Current config: {repr(cv_config_)}")

        current_run_config = deepcopy(run_config)
        for k, v in cv_config_.items():
            k, param = k.split('__')
            current_run_config[k][param] = v

        for fold, (train_idx, val_idx) in (
            enumerate(kf.split(dataset["train"]), 1)
        ):
            logger.info("Fold %d.", fold)

            torch.cuda.reset_peak_memory_stats()

            base_model = create_model(current_run_config)
            adapter_config = create_adapter_config(current_run_config, "l1ra")

            train_dataset = dataset["train"].select(train_idx)
            val_dataset = dataset["train"].select(val_idx)

            fold_dataset = DatasetDict(
                {"train": train_dataset, "test": val_dataset}
            )

            report = train_and_evaluate(
                base_model,
                tokenizer,
                adapter_config,
                fold_dataset,
                current_run_config
            )
            fold_reports.append(report)
            del base_model
            gc.collect()
            torch.cuda.empty_cache()

        ppl_values = np.array([r["ppl"] for r in fold_reports])
        param_values = np.array([r["adapter_params"] for r in fold_reports])

        ppl_mean = ppl_values.mean()
        ppl_std = ppl_values.std()
        param_mean = param_values.mean()
        param_std = param_values.std()

        logger.info(f"{K} folds completed.")
        logger.info(f"Perplexity: mean={ppl_mean}, std={ppl_std}")
        logger.info(f"Params: mean={param_mean}, std={param_std}")
        cv_report.append((*cv_config_.values(), ppl_mean, ppl_std, param_mean, param_std))

        df = pd.DataFrame(
            cv_report,
            columns=[*cv_config_.keys(), "ppl_mean", "ppl_std", "param_mean", "param_std"]
        )
        df.to_csv(os.path.join(directory, "cross-validation.csv"), index=False)

    logger.info(f"Cross-validation report saved at \"{directory}\"")


@click.command()
@click.option('--config-path', help='Path of training/cv config file.')
def main(config_path):
    config = load_config(config_path)
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"

    if "cv_k" in config:
        configs = [load_config(c) for c in config["run_cv_on"]]
        for run in configs:
            cross_validation(config, run)

    else:
        tokenizer = load_tokenizer(config)
        dataset = load_and_preprocess_dataset(config, tokenizer)

        for adapter_type in config["to_train"]:
            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
            # Train
            adapter_config = create_adapter_config(config, adapter_type, dataset=dataset)
            model = create_model(config, adapter_config)
            report = train_and_evaluate(
                model,
                tokenizer,
                adapter_config,
                dataset,
                config
            )
            save_report(adapter_type, report, config_path)
            # Clear model after training
            del model
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
    print('DeepSpeed enabled:', is_deepspeed_zero3_enabled())

    main()
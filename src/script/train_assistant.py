import logging
import os
import time
import gc
from datetime import datetime
from shutil import copy2
import click
import math
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import GradScaler
import yaml
from datasets import DatasetDict, load_dataset
from peft import AdaLoraConfig, LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel
from sklearn.model_selection import KFold
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizer
)
import torchmetrics

from l1ra import L1RAConfig

from bitsandbytes.optim import PagedAdamE32bit

from typing import Dict, Union, Optional, Tuple, List


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


def load_config(path: str) -> Dict:
    with open(path, "r") as file:
        config = yaml.safe_load(file)

    return config


def load_guanaco(config: Dict, tokenizer: PreTrainedTokenizer, validation_split: float = 0.1) -> DatasetDict:
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


def load_open_orca(
        config: Dict, tokenizer: PreTrainedTokenizer, validation_split: float = 0.1, test_split: float = 0.1
) -> DatasetDict:
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


def load_and_preprocess_dataset(config: Dict, tokenizer: PreTrainedTokenizer) -> DatasetDict:
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


def get_collate(tokeniser: PreTrainedTokenizer):
    def collate(batch):
        input_encodings = tokeniser(
            [sample['text'] for sample in batch],
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        input_encodings['labels'] = input_encodings['input_ids'].clone()
        input_encodings['labels'][~input_encodings['input_ids'].bool()] = -100

        return input_encodings

    return collate


def create_model(config: Dict, adapter_config: Union[LoraConfig, AdaLoraConfig, L1RAConfig]) -> PeftModel:
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
        token=token,
        attn_implementation='flash_attention_2'
    ).to('cuda:0')
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    logger.info("%s loaded.", model_id)

    model = get_peft_model(model, adapter_config)

    return model


def create_adapter_config(
        config: Dict, adapter_type: str, dataset: Optional[DatasetDict] = None
) -> Union[LoraConfig, AdaLoraConfig, L1RAConfig]:
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


def compute_n_adapter_params(model: PeftModel) -> int:
    params = 0

    for n, p in model.named_parameters():
        if "lora" in n:
            params += p.numel()

    return params


def rank_evolution(model: PeftModel, model_id: str, training_steps) -> pd.DataFrame:
    model = model.base_model
    rank_evolution = model.rank_evolution

    n_layers = AutoConfig.from_pretrained(model_id).num_hidden_layers
    model_shape = (n_layers, len(rank_evolution[0])//n_layers)
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


def create_optimiser(
        model: PeftModel, dataloader: torch.utils.data.DataLoader, config: Dict, adapter_type: str
) -> Tuple[PagedAdamE32bit, Optional[torch.optim.lr_scheduler.OneCycleLR]]:
    if adapter_type in ('lora', 'adalora'):
        optimiser = PagedAdamE32bit(
            [p for n, p in model.named_parameters() if 'lora' in n and 'lora_c' not in n],
            lr=config['training_args']['learning_rate'],
            lasso=0.0,
            weight_decay=config['training_args'].get('weight_decay', 0.01)
        )
    elif adapter_type in ('l1ra',):
        optimiser = PagedAdamE32bit(
            [
                {
                    'params': [p for n, p in model.named_parameters() if 'lora_c' in n],
                    'weight_decay': 0.0,
                    'lr': model.peft_config["default"].eta_c
                },
                {'params': [p for n, p in model.named_parameters() if 'lora' in n and 'lora_c' not in n], 'lasso': 0.0}
            ],
            lr=config['training_args']['learning_rate'],
            lasso=model.peft_config["default"].l1ra_lambda,
            weight_decay=config['training_args'].get('weight_decay', 0.01)
        )
    else:
        raise ValueError()

    lr_scheduler = None
    if config['training_args'].get('lr_scheduler_type') is not None:
        training_steps = int(math.ceil(
            len(dataloader) / (
                config['training_args'].get('per_device_train_batch_size', 1) *
                config['training_args'].get('gradient_accumulation_steps', 1)
            ))
        )
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimiser,
            [p['lr'] for p in optimiser.param_groups],
            training_steps,
            pct_start=config['training_args'].get('warmup_ratio', 0.0)
        )
        if adapter_type == 'l1ra':
            optimiser.param_groups[0]['lr'] = model.peft_config["default"].eta_c

    return optimiser, lr_scheduler


def restart_optimiser(
        optimiser: PagedAdamE32bit,
        lr_scheduler: Optional[torch.optim.lr_scheduler.OneCycleLR],
        model: PeftModel,
        dataloader: DatasetDict,
        config: Dict,
        adapter_type: str
) -> Tuple[PagedAdamE32bit, Optional[torch.optim.lr_scheduler.OneCycleLR]]:
    learning_rates = [p['lr'] for p in optimiser.param_groups]
    optimiser = create_optimiser(model, dataloader, config, adapter_type)
    for p, lr in zip(optimiser.param_groups, learning_rates):
        p['lr'] = lr

    if lr_scheduler is not None:
        lr_scheduler.optimizer = optimiser

    return optimiser, lr_scheduler


def train(
        model: PeftModel,
        dataloader: torch.utils.data.DataLoader,
        optimiser: PagedAdamE32bit,
        lr_scheduler: Optional[torch.optim.lr_scheduler.OneCycleLR],
        config: Dict,
        adapter_type: str
) -> List[Dict]:
    model.train()

    max_grad_norm = config['training_args'].get('max_grad_norm')
    scaler = GradScaler() if config['training_args'].get('bf16', True) else None

    total_steps = config['training_args'].get('num_train_epochs') * int(
        math.ceil(len(dataloader) / config['training_args'].get('gradient_accumulation_steps', 1))
    )
    warmup_steps = int(math.ceil(total_steps * config['training_args'].get('warmup_ratio', 0.0)))
    step = 0

    history = []
    training_loss = None

    start = time.time()

    for epoch in range(config['training_args'].get('num_train_epochs', 1)):
        for i, batch in enumerate(tqdm(dataloader, desc="Training")):
            if scaler is not None:
                with torch.autocast(device_type=torch.device('cuda:0').type, dtype=torch.bfloat16):
                    output = model(**batch.to(model.device))
                    loss = output.loss / config['training_args'].get('gradient_accumulation_steps', 1)
                scaler.scale(loss).backward()
            else:
                output = model(**batch.to(model.device))
                loss = output.loss / config['training_args'].get('gradient_accumulation_steps', 1)
                loss.backward()
            training_loss = (training_loss + loss.detach()) if training_loss is not None else loss.detach()
            if (i + 1) % config['training_args'].get('gradient_accumulation_steps', 1) == 0 or i + 1 == len(dataloader):
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(optimiser.parameters(), max_grad_norm)
                if scaler is not None:
                    scaler.step(optimiser)
                    scaler.update()
                else:
                    optimiser.step()
                if adapter_type == 'adalora':
                    model.base_model.update_and_allocate(step)
                if adapter_type == 'l1ra':
                    updated = model.update_ranks(step, total_steps, warmup_steps)
                    if updated:
                        optimiser, lr_scheduler = restart_optimiser(
                            optimiser, lr_scheduler, model, dataloader, config, adapter_type
                        )
                optimiser.zero_grad()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                    if adapter_type == 'l1ra':
                        optimiser.param_groups[0]['lr'] = model.peft_config["default"].eta_c
                if (step + 1) % int(total_steps * config['training_args'].get('logging_steps', 1.0)) or (
                        epoch + 1 == config['training_args'].get('num_train_epochs', 1) and i + 1 == len(dataloader)
                ):
                    history.append({
                        'step': step,
                        'training_loss': training_loss.item(),
                        'epoch': epoch + (i + 1 / len(dataloader)),
                        'elapsed_time': time.time() - start,
                    } | {
                        f'lr_param_group_{i}': p['lr'] for i, p in enumerate(optimiser.param_groups)
                    })
                training_loss = None
                step += 1

    return history


@torch.no_grad()
def eval(model: PeftModel, dataloader: torch.utils.data.DataLoader) -> Tuple[float, float]:
    loss = []
    metric = torchmetrics.text.Perplexity(ignore_index=-100).to(model.device)

    for batch in tqdm(dataloader, desc="Evaluation"):
        output = model(**batch.to(model.device))
        loss.append(output.loss)
        metric(output.logits[:, :-1], batch.labels[:, 1:])

    loss = torch.cat(loss).mean().cpu().item()
    ppl = metric.compute()

    return loss, ppl

def train_and_evaluate(
        model: PeftModel, tokeniser: PreTrainedTokenizer, adapter_config: Dict, dataset, config: Dict, adapter_type: str
):
    train_dataloader = torch.utils.data.DataLoader(
        dataset['train'],
        batch_size=config['training_args'].get('per_device_train_batch_size', 1),
        shuffle=True,
        collate_fn=get_collate(tokeniser),
        num_workers=8
    )
    optimiser, lr_scheduler = create_optimiser(model, train_dataloader, config, adapter_type)

    start = time.time()

    history = train(model, train_dataloader, optimiser, lr_scheduler, config, adapter_type)

    end = time.time()

    history = pd.DataFrame(history)
    adapter_params = compute_n_adapter_params(model)
    peak_mem_usage = torch.cuda.max_memory_allocated()
    time_taken = end - start

    logger.info("Model successfully trained.")

    logger.info("%s", [p for n,p in model.named_parameters() if "lora_c" in n])

    model.eval()
    if isinstance(adapter_config, L1RAConfig):
        logger.info("l1ra lambda: %f", model.peft_config["default"].l1ra_lambda)

    test_dataloader = torch.utils.data.DataLoader(
        dataset['test'],
        batch_size=config['training_args'].get('per_device_eval_batch_size', 1),
        collate_fn=get_collate(tokeniser),
        num_workers=8
    )

    test_loss, test_ppl = eval(model, test_dataloader)

    logger.info("Test loss: %f", test_loss)
    logger.info("Test perplexity: %f", test_ppl)

    report = {
        "history": pd.DataFrame(history),
        "adapter_params": adapter_params,
        "peak_mem_usage": peak_mem_usage,
        "time_taken": time_taken,
        "ppl": test_ppl
    }

    if isinstance(adapter_config, L1RAConfig):
        report["rank_evolution"] = rank_evolution(model, config["model_id"], history['step'].values[-1])

    return report


def save_report(adapter_type: str, report: Dict, config_file_path: str):
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


def load_tokenizer(config: Dict) -> PreTrainedTokenizer:
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


@click.command()
@click.option('--config-path', help='Path of training/cv config file.')
def main(config_path):
    config = load_config(config_path)
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"

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
            config,
            adapter_type
        )
        save_report(adapter_type, report, config_path)
        # Clear model after training
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

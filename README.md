# L1RA: L1-Regularised Rank Assignment in LoRA Fine-Tuning
L1RA: L1-Regularised Rank Assignment in LoRA Fine-Tuning is a method that dynamically reassigns LoRA ranks during fine-tuning.  This is the repository of MSc thesis held at Politecnico di Milano in Computer Science and Engineering.

L1RA automatically prunes and reassign ranks during the training process of a model. This allows the model to better optimize its rank distribution, instead of being constant like in LoRA. L1RA is aimed to have almost no discernible impact on training time and memory.

## Basic Usage

1. Import `L1RAConfig` and `L1raTrainer` from this module. `L1RACofig` is going to replace the usual `LoraConfig` while `L1RATrainer` is a `SFTTrainer` sublcass from package `trl`.

```python
from l1ra import L1RAConfig, L1RATrainer
```

2. Create a config just as you would do with `LoraConfig`. Here is an example:

```python
config = L1RAConfig(
    task_type=peft.TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules="all-linear",
    bias="none",
)
```

3. You also need to load your model of choiche, tokenizer and dataset. Here is an example:

```python
from transformers import AutoModelForCausalLM
from datasets import load_dataset

model_id = "your model of choice"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

dataset = load_dataset("your dataset")
```

4. To train the model you follow the normal procedure you would do to train using the `SFTTrainer` (or just the plain Hugging Face `Trainer`). First create the `TrainingArguments` class and then pass it to the `L1RATrainer` instance. You can directly pass the `L1RAConfig` to the trainer instead of calling the `get_peft_model()` method, because the trainer will automatically take care of it. Additionally, the trainer can automatically tokenize your input. Here is an example:

```python
from transformers import TrainingArguments, DataCollatorForLanguageModeling

training_args = TrainingArguments(
    output_dir="trainer_output/",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_checkpointing=True,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.1,
    num_train_epochs=1,
    bf16=True,
    max_grad_norm=0.3,
    optim="paged_adamw_8bit",
)

trainer = L1RATrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    dataset_text_field="text",
    peft_config=config,
    eval_dataset=dataset["validation"],
    max_seq_length=512,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    tokenizer=tokenizer
)

trainer.train()
```

# L1RA: L1-Regularised Rank Assignment in LoRA Fine-Tuning

L1RA: L1-Regularised Rank Assignment in LoRA Fine-Tuning is a method that dynamically reassigns LoRA ranks during fine-tuning.  This is the repository of MSc thesis held at Politecnico di Milano in Computer Science and Engineering.

L1RA automatically prunes and reassign ranks during the training process of a model. This allows the model to better optimize its rank distribution, instead of being constant like in LoRA. L1RA is aimed to have almost no discernible impact on training time and memory.

## Basic Usage

1. Install the `l1ra` package:

```sh
pip install git+https://github.com/raul-singh/L1RA.git
```

2. Import `L1RAConfig` and `L1raTrainer` from this module. `L1RACofig` is going to replace the usual `LoraConfig` while `L1RATrainer` is a `SFTTrainer` sublcass from package `trl`.

```python
from l1ra import L1RAConfig, L1RATrainer
```

3. Create a config just as you would do with `LoraConfig`. Here is an example:

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

4. You also need to load your model of choice, tokenizer and dataset. Here is an example:

```python
from transformers import AutoModelForCausalLM
from datasets import load_dataset

model_id = "your model of choice"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

dataset = load_dataset("your dataset")
```

5. To train the model you follow the normal procedure you would do to train using the `SFTTrainer` (or just the plain Hugging Face `Trainer`). First create the `TrainingArguments` class and then pass it to the `L1RASFTTrainer` instance. You can directly pass the `L1RAConfig` to the trainer instead of calling the `get_peft_model()` method, because the trainer will automatically take care of it. Additionally, the trainer can automatically tokenize your input. Here is an example:

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
    save_strategy="no",
)

trainer = L1RASFTTrainer(
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

NOTE: as of right now, L1RA does not support saving and loading the model, so it's mandatory to have `save_strategy="no"` when defining `TrainingArguments`.

There is also `L1RATrainer` available, which is an extension of the basic `Trainer` from the `transformers` library. The interface is the same, and it can be used whenever the `L1RASFTTraner` does not fit the job.

### Optimizer Note

L1RA makes use of the experimental `AdamE` optimizer. As of right now, `L1RATrainer` and `L1RASFTTrainer` will force the use of `AdamE` when specifying another AdamW optimizer like `adamw` or `adamw-torch`.

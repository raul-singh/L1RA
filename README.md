# L1RA: L1-Regularised Rank Assignment in LoRA Fine-Tuning

L1RA: L1-Regularised Rank Assignment in LoRA Fine-Tuning is a method that dynamically reassigns LoRA ranks during fine-tuning.  

L1RA automatically prunes and reassigns ranks during the training process of a model. 
This allows the model to better optimize its rank distribution, instead of being constant like in LoRA. 
L1RA is aimed to have almost no discernible impact on training time and memory.

We presented L1RA in a [research paper](https://arxiv.org/abs/2509.04884) at [ICNLSP 2025](https://www.icnlsp.org/2025welcome/).
To cite our work, please refer to the [Section References](#references)
The original project was developed as an [M.Sc. Thesis](https://www.politesi.polimi.it/handle/10589/223901) in Computer Science and Engineering at Politecnico di Milano.

L1RA includes the [🍦 Memory-GELATO ](https://github.com/raul-singh/memory-gelato) submodule to estimate training hyperparameters to fit in the GPU memory given the LLM, the data and the GPU specifications.

## Basic Usage

> [!WARNING]  
> Before using L1RA make sure to have a proper environment installed. 
> Refer to [Section "Build environment"](#build-environment) for further details.

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

> [!NOTE]  
> As of right now, L1RA does not support saving and loading the model, so it's mandatory to have `save_strategy="no"` when defining `TrainingArguments`.

There is also `L1RATrainer` available, which is an extension of the basic `Trainer` from the `transformers` library. The interface is the same, and it can be used whenever the `L1RASFTTraner` does not fit the job.

## Build environment
  
> [!WARNING]  
> Run all the scripts and the commands mentioned in this section **from the main repository directory**.

As for now, L1RA requires a custom environment to run properly.
We provide a script to install a conda environment to run our code.
Make sure you have:
- a CUDA compatible GPU 
- [Anaconda](https://www.anaconda.com/) installed, you can get it from https://www.anaconda.com/download/ 
- building tools installed, you can install them running:
  ```bash
  apt install -y build-essential cmake
  ```
  
### Main environment

Run the following script to build a conda environment named `l1ra` with CUDA 12.2:

```bash
bash ./build_env.sh
```

To show all the available options, run:

```bash
bash ./build_env.sh -h
```

In case of problems with `triton` see this issue: https://github.com/bitsandbytes-foundation/bitsandbytes/issues/328#issuecomment-2660001212

### Baselines environment

To replicate the baseline experiments, you will need a separate environment.
Run the following script **from the main repository directory** to build a conda environment named `l1ra_baseline` with CUDA 12.2:

```bash
bash ./build_env_baselines.sh
```

To show all the available options, run:

```bash
bash ./build_env_baselines.sh -h
```

> [!NOTE]  
> This part is necessary only to replicate the experiments and is not required to use L1RA.

## Run experiments
  
> [!WARNING]  
> Run all the scripts and the commands mentioned in this section **from the main repository directory**.

To (re-)run the paper experiments use the dedicated script with this command

```bash
bash ./run_experiments.sh
```

To run it in background use the command

```bash
nohup bash ./run_experiments.sh > experiments_"$(date '+%Y_%m_%d_%H_%M_%S')".out &
```

> [!NOTE]  
> The script expects that you build the main and baseline environments using the default names.

## References

BibTeX entry to cite our work:

```bibtex
@inproceedings{singh-etal-2025-l1ra,
    title = "L1RA: Dynamic Rank Assignment in LoRA Fine-Tuning",
    author = "Singh, Raul  and
      Brunello, Nicol{\`o}  and
      Scotti, Vincenzo  and
      Carman, Mark",
    booktitle = "Proceedings of the 8th International Conference on Natural Language and Speech Processing (ICNLSP 2025)",
    month = aug,
    year = "2025",
    address = "Odense, Denmark",
    publisher = "Association for Computational Linguistics"
}
```

## Acknowledgements

- Raul Singh: ([raul.singh@mail.polimi.it](mailto:raul.singh@mail.polimi.it))
- Nicolò Brunello ([nicolo.brunello@polimi.it](mailto:nicolo.brunello@polimi.it))
- Vincenzo Scotti: ([vincenzo.scotti@kit.edu](mailto:vincenzo.scotti@kit.edu))
- Mark James Carman: ([mark.carman@polimi.it](mailto:mark.carman@.polimi.it))

import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, AdaLoraConfig, prepare_model_for_kbit_training, get_peft_model
from l1ra import L1RAConfig
from bitsanddytes import PagedAdamW32bit, PagedAdamE32bit


models = ['mistralai/Mistral-7B-v0.3', 'meta-llama/Meta-Llama-3.1-8B']
adapters_r = {'lora': [16], 'adalora': [24, 16], 'l1ra': [16]}
batch_and_accumulation = {
    'mistralai/Mistral-7B-v0.3': {'batch_size': 8, 'accumulation_steps': 2},
    'meta-llama/Meta-Llama-3.1-8B': {'batch_size': 4, 'accumulation_steps': 4}
}


for model in models:
    print('model:', model)
    # Load tokeniser
    tokeniser = AutoTokenizer.from_pretrained(model)
    tokeniser.pad_token = tokeniser.eos_token
    for adapter, ranks in adapters_r.items():
        print('adapter:', adapter)
        for r in ranks:
            print('rank:', r)
            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
            # Load model
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            transformer = AutoModelForCausalLM.from_pretrained(
                model,
                device_map='cuda:0',
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16
            )
            transformer.config.use_cache = False
            transformer.gradient_checkpointing_enable()
            transformer = prepare_model_for_kbit_training(transformer)
            # Load adapters
            if adapter == 'lora':
                lora_config = LoraConfig(
                    r=16,
                    lora_alpha=16,
                    lora_dropout=0.1,
                    target_modules='all-linear',
                    bias=None,
                    task_type='CAUSAL_LM'
                )
            elif adapter == 'adalora':
                lora_config = AdaLoraConfig(
                    r=16,
                    target_r=16,
                    init_r=24,
                    lora_alpha=16,
                    lora_dropout=0.1,
                    target_modules='all-linear',
                    bias=None,
                    task_type='CAUSAL_LM'
                )
            elif adapter == 'l1ra':
                lora_config = L1RAConfig(
                    r=16,
                    lora_alpha=16,
                    lora_dropout=0.1,
                    target_modules='all-linear',
                    bias=None,
                    task_type='CAUSAL_LM',
                    l1ra_lambda=0.005,
                    eta_c=0.02,
                    rank_update_ratio=0.05
                )
            else:
                raise ValueError()
            transformer = get_peft_model(transformer, lora_config)
            # Generate fake data
            input_encodings = tokeniser(
                [tokeniser.eos_token * 1024] * batch_and_accumulation[model]['batch_size'],
                return_tensors='pt',
                padding=True
            ).to('cuda:0')
            # Create optimiser
            if adapter == 'lora' or adapter == 'adalora':
                optimiser = PagedAdamW32bit(
                    [p for n, p in transformer.named_parameters() if 'lora' in n]
                )
            elif adapter == 'l1ra':
                optimiser = PagedAdamE32bit(
                    [
                        {
                            "params": [p for n, p in transformer.named_parameters() if 'lora_c' in n],
                            "weight_decay": 0.0
                        },
                        {
                            "params": [p for n, p in transformer.named_parameters() if
                                       'lora' in n and 'lora_c' not in n],
                            "lasso": 0.0,
                        }
                    ]
                )
            else:
                raise ValueError()
            # Simulate forward step
            if adapter == 'lora':
                for _ in range(batch_and_accumulation[model]['accumulation_steps']):
                    output = transformer(**input_encodings, labels=input_encodings.input_ids)
                    loss = output.loss / batch_and_accumulation[model]['accumulation_steps']
                    loss.backward()
                optimiser.step()
                optimiser.zero_grad()
            elif adapter == 'adalora':
                for _ in range(batch_and_accumulation[model]['accumulation_steps']):
                    output = transformer(**input_encodings, labels=input_encodings.input_ids)
                    loss = output.loss / batch_and_accumulation[model]['accumulation_steps']
                    loss.backward()
                optimiser.step()
                transformer.base_model.update_and_allocate(1000)
                optimiser.zero_grad()
            elif adapter == 'l1ra':
                for ...:
                    for _ in range(batch_and_accumulation[model]['accumulation_steps']):
                        output = transformer(**input_encodings, labels=input_encodings.input_ids)
                        loss = output.loss / batch_and_accumulation[model]['accumulation_steps']
                        loss.backward()
                    optimiser.step()
                    optimiser.zero_grad()
            else:
                raise ValueError()
            # Get measures
            peak_mem_usage = torch.cuda.max_memory_allocated()
            print('memory:', peak_mem_usage / (1024 ** 3))
            # Clear model after training
            del transformer
            gc.collect()
            torch.cuda.empty_cache()

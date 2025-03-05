from memory_gelato import MemoryArgs, simple_peak_estimation


models = ['mistralai/Mistral-7B-v0.3', 'meta-llama/Meta-Llama-3.1-8B']
adapters_r = {'lora': [16], 'adalora': [24, 16], 'l1ra': [16]}


for model in models:
    print('model:', model)
    for adapter, ranks in adapters_r.items():
        print('adapter:', adapter)
        for r in ranks:
            print('rank:', r)
            memory_args = MemoryArgs(
                model_id = model,
                batch_size=8 if 'mistral' in model else 4,
                gradient_accumulation=2 if 'mistral' in model else 4,
                max_sequence_length=1024,
                adapter_type=adapter,
                adapter_rank=r,
                bnb_quantization_bit=4
            )
            memory_estimate = simple_peak_estimation(memory_args)
            print('memory estimate:', memory_estimate / (1024 ** 3))
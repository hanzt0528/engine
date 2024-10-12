## Dowload models
```
examples/gpt-2/models# git clone https://hf-mirror.com/cerebras/Cerebras-GPT-111M
Cloning into 'Cerebras-GPT-111M'...
remote: Enumerating objects: 61, done.
remote: Counting objects: 100% (61/61), done.
remote: Compressing objects: 100% (27/27), done.
remote: Total 61 (delta 33), reused 61 (delta 33), pack-reused 0 (from 0)
Unpacking objects: 100% (61/61), done.
```

## Convert to ggml
```
/examples/gpt-2#conda activate ggml
/examples/gpt-2#python convert-cerebras-to-ggml.py models/Cerebras-GPT-111M/
{'model_type': 'gpt2', 'attn_pdrop': 0.0, 'scale_attn_weights': True, 'resid_pdrop': 0.0, 'n_inner': 3072, 'n_embd': 768, 'layer_norm_epsilon': 1e-05, 'n_positions': 2048, 'activation_function': 'gelu', 'n_head': 12, 'n_layer': 10, 'tie_word_embeddings': True, 'vocab_size': 50257, 'embd_pdrop': 0.0}
Processing variable: transformer.wte.weight with shape:  (50257, 768)
  Converting to float16
...
...
Processing variable: lm_head.weight with shape:  (50257, 768)
  Converting to float16
Done. Output file: models/Cerebras-GPT-111M//ggml-model-f16.bin
```
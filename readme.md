# Language Model Emergent Ability Distillation. 

Notebooks 
* `flan_t5_gsm8k.ipynb`: run GSM8K on Flan-T5
* `flan_t5_gsm8k_verifier.ipynb`: run GSM8K on Flan-T5 with verifier
* `flan_t5_gsm8k_vis.ipynb`: visualization of decoding probability of Flan-T5
* `flan_t5_mmlu.ipynb`: run mmlu on Flan-T5
* `flan_t5_multiarith.ipynb`: run multiarith on Flan-T5
* `opt_gsm8k.ipynb`: run multiarith on OPT (have not gone through this yet)

Scripts
* `codex_decode_gsm8k.py`: decode gsm8k training set with codex
* `flan_t5_decode_gsm8k.py`: decode gsm8k training set with Flan-T5
* `flan_t5_verifier_decode_gsm8k.py`: decode gsm8k training set with Flan-T5 + verifier (TBC)

Distillation 
* `train_distill_t5.py`: train the distillation algorithm 
* `trainer_distill.py`: trainer for emergent ability distillation 


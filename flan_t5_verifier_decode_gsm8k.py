"""
Decode GSM8K training data using the T5 model + verifier.

nohup python -u flan_t5_verifier_decode_gsm8k.py\
  --gpu_id 2\
  --output_path outputs/gsm8k/train_flan_t5_verifier_complex.txt\
  --debug 1\
  --num_sample 50\
  &> logs/flan_t5_verifier_decode_gsm8k.log &

tail -f logs/flan_t5_verifier_decode_gsm8k.log
"""


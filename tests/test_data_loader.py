import numpy as np 

from transformers import T5Tokenizer
from src.data_utils import GSM8KCodexAugmentedDataset

# %load_ext autoreload
# %autoreload 2

tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-xxl')
dataset = GSM8KCodexAugmentedDataset()
batch = dataset.batches[0]
procesed_batch = dataset.process_batch(tokenizer, batch) 
dataset.decode_batch(batch, tokenizer)
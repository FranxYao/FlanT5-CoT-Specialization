"""
Decode GSM8K training data using the T5 model.
"""

import datasets
import torch
import re
import numpy as np

from tqdm import tqdm
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration

def flan_decode(model, input):
  input_ids = tokenizer(prompt_q, return_tensors="pt").input_ids.to("cuda:0")
  return 

def main():
  # load the dataset
  gsm8k = load_dataset('gsm8k', 'main')

  # load the model
  tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
  model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl").to('cuda:1')

  # decode the dataset
  return 

if __name__ == '__main__':
  main()
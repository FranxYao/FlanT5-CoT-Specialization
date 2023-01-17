"""
Decode GSM8K training data using the T5 model.
TODO: adaptive batch size, such that max_len * batch_size = const 
"""

import time 
import torch
import re
import argparse
import os
import pytz 
import hydra
import json

import numpy as np
import torch.nn.functional as F

from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from omegaconf import DictConfig, OmegaConf
from src.utils import tprint, parse_pred_ans
from test_distill_multiple import load_test_data

GSM8K_VALIDATION_INDEX_PATH = 'lib_prompt/validation_index.npy'
MULTIARITH_PATH = 'data/multiarith/MultiArith.json'
MULTIARITH_VALIDATION_INDEX_PATH = 'data/multiarith/validation_index.npy'


# def load_test_data(test_data):
#     # TODO: add multiarith/ other math datasets
#     if(test_data == 'gsm8k_dev'):
#         gsm8k = load_dataset('gsm8k', 'main')
#         validation_index = np.load(GSM8K_VALIDATION_INDEX_PATH)
#         data = gsm8k['train'].select(validation_index)
#         data_ = []
#         for q, a in zip(data['question'], data['answer']): 
#             data_.append({'question': q, 'answer': a})
#     elif(test_data == 'gsm8k_test'):
#         gsm8k = load_dataset('gsm8k', 'main')
#         data = gsm8k['test']
#         data_ = []
#         for q, a in zip(data['question'], data['answer']): 
#             data_.append({'question': q, 'answer': a})
#     elif(test_data == 'multiarith_test'):
#         dataset = json.load(open(MULTIARITH_PATH))
#         dev_ind = np.load(MULTIARITH_VALIDATION_INDEX_PATH)
#         # dev_data = [dataset[i] for i in dev_ind]
#         test_data = [d for i, d in enumerate(dataset) if i not in dev_ind]
#         data_ = []
#         for d in test_data:
#             data_.append({'question': d['sQuestion'][1:-1], 'answer': d['lSolutions']})
#     else:
#         raise ValueError('Invalid test data: %s' % test_data)
#     return data_


@hydra.main(version_base=None, config_path="src/conf", config_name="config_inference")
def main(args : DictConfig):
    print(OmegaConf.to_yaml(args))

    # load the dataset
    dataset = load_test_data(args.test_data)

    # load the model
    tprint('Loading the model ... ')
    start_time = time.time()
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
    model = T5ForConditionalGeneration.from_pretrained(args.base_model)
    if(args.model_size == '11b'):
        model.parallelize(args.device_map)
    else:
        model.to('cuda:' + str(args.gpu_id))

    tprint('Model loaded in %.1f seconds.' % (time.time() - start_time))

    # load the prompt
    prompt = open(args.prompt_path).read()

    # decode the dataset
    tprint('Start decoding ... ')
    i = 0
    output_path = args.output_path + args.test_data + '_' + args.base_model.split('/')[-1] + '.txt'
    tprint('Model output to: %s' % output_path)

    # TODO: change this to batch version
    with open(output_path, 'w') as fd:
        tqdm_total = len(dataset) // args.batch_size
        if(len(dataset) % args.batch_size != 0): tqdm_total += 1
        for i in tqdm(range(0, len(dataset), args.batch_size), total=tqdm_total):
            questions = []
            q_batch = []
            a_batch = []
            for k in range(args.batch_size):
                if(i + k >= len(dataset)): break
                
                q = dataset[i + k]['question']
                q_batch.append(q)
                a = dataset[i + k]['answer']
                a_batch.append(a)
                
                prompt_q = prompt + '\nQ: ' + q + '\n'
                prompt_q += "Let's think step by step\n"
                questions.append(prompt_q)
                
            inputs = tokenizer(questions, padding=True, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(inputs['input_ids'].to(model.device), 
                                         attention_mask=inputs['attention_mask'].to(model.device), 
                                         max_length=256
                                         )
            
            for q, a, ans_ in zip(q_batch, a_batch, outputs):
                ans_ = tokenizer.decode(ans_).replace('<pad>', '').strip()
                fd.write('Q: %s\nA_model:\n%s\nA:\n%s\n\n' % (q, ans_, a))

    _, _, _ = parse_pred_ans(output_path)
    return 

if __name__ == '__main__':
  main()
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
import pickle

import numpy as np
import torch.nn.functional as F

from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from omegaconf import DictConfig, OmegaConf
from src.utils import tprint, parse_pred_ans

GSM8K_VALIDATION_INDEX_PATH = 'lib_prompt/validation_index.npy'
MULTIARITH_PATH = 'data/multiarith/MultiArith.json'
MULTIARITH_VALIDATION_INDEX_PATH = 'data/multiarith/validation_index.npy'
ASDIV_QUESTIONS_PATH = 'processed_data/ASDiv_questions.pkl'
ASDIV_ANSWERS_PATH = 'processed_data/ASDiv_answers.pkl'
SVAMP_PATH = 'data/SVAMP/SVAMP.json'


def load_test_data(test_data):
    # TODO: add multiarith/ other math datasets
    if(test_data == 'gsm8k_dev'):
        gsm8k = load_dataset('gsm8k', 'main')
        validation_index = np.load(GSM8K_VALIDATION_INDEX_PATH)
        data = gsm8k['train'].select(validation_index)
        data_ = []
        for q, a in zip(data['question'], data['answer']): 
            data_.append({'question': q, 'answer': a})
    elif(test_data == 'gsm8k_test'):
        gsm8k = load_dataset('gsm8k', 'main')
        data = gsm8k['test']
        data_ = []
        for q, a in zip(data['question'], data['answer']): 
            data_.append({'question': q, 'answer': a})
    elif(test_data == 'multiarith_test'):
        dataset = json.load(open(MULTIARITH_PATH))
        dev_ind = np.load(MULTIARITH_VALIDATION_INDEX_PATH)
        # dev_data = [dataset[i] for i in dev_ind]
        test_data = [d for i, d in enumerate(dataset) if i not in dev_ind]
        data_ = []
        for d in test_data:
            data_.append({'question': d['sQuestion'][1:-1], 'answer': d['lSolutions']})
    elif(test_data == 'asdiv_test'):
        questions = pickle.load(open(ASDIV_QUESTIONS_PATH, 'rb'))
        answers = pickle.load(open(ASDIV_ANSWERS_PATH, 'rb'))
        data_ = []
        for q, a in zip(questions, answers):
            data_.append({'question': q, 'answer': a})
    elif(test_data == 'svamp_test'):
        data = json.load(open(SVAMP_PATH))
        data_ = []
        for d in data:
            data_.append({'question': d['Body'] + '. ' + d['Question'], 
                          'answer': str(d['Answer'])}
                          )
    else:
        raise ValueError('Invalid test data: %s' % test_data)
    return data_

def test_model(dataset, tokenizer, model, base_prompt, args, model_dir):
    # decode the dataset
    tprint('Start decoding ... ')
    i = 0
    output_path = args.output_path + args.test_data + '_' + model_dir.split('/')[-1] + '.txt'
    tprint('Model output to: %s' % output_path)

    # TODO: change this to batch version
    if(isinstance(args.batch_size, int)):
        batch_size = args.batch_size
    else:
        batch_size = args.batch_size[args.test_data.split('_')[0]]
    with open(output_path, 'w') as fd:
        tqdm_total = len(dataset) // batch_size
        if(len(dataset) % batch_size != 0): tqdm_total += 1
        for i in tqdm(range(0, len(dataset), batch_size), total=tqdm_total):
            questions = []
            q_batch = []
            a_batch = []
            for k in range(batch_size):
                if(i + k >= len(dataset)): break
                
                q = dataset[i + k]['question']
                q_batch.append(q)
                a = dataset[i + k]['answer']
                a_batch.append(a)

                if(args.prompt_mode == 'zero_shot_cot'):
                    prompt_q = base_prompt + q + '\n' + "Let's think step by step\n"
                elif(args.prompt_mode == 'zero_shot'):
                    prompt_q = base_prompt + q + '\n' + 'A: '
                elif(args.prompt_mode == 'cot_4_cases'):
                    prompt_q = base_prompt + q + '\n' + "Let's think step by step\n"
                elif(args.prompt_mode == 'answer_only_4_cases'):
                    raise ValueError('Invalid prompt mode: %s' % args.prompt_mode)
                    prompt_q = base_prompt + q + '\n' + "A: " # TODO: remove CoT in this case
                else: 
                    raise ValueError('Invalid prompt mode: %s' % args.prompt_mode)

                # import ipdb; ipdb.set_trace()
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


@hydra.main(version_base=None, config_path="src/conf", config_name="config_inference_multiple")
def main(args : DictConfig):
    print(OmegaConf.to_yaml(args))

    # load the dataset
    dataset = load_test_data(args.test_data)
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer)

    if(args.prompt_mode == 'zero_shot_cot'):
        base_prompt = 'Q: '
    elif(args.prompt_mode == 'cot_4_cases'):
        base_prompt = open(args.prompt_path).read() + '\nQ: '
    else: 
        raise ValueError('Invalid prompt mode: %s' % args.prompt_mode)

    for i in args.iter:
        start_time = time.time()
        model_dir = args.base_model + str(i)
        tprint('Loading the model from %s' % model_dir)
        model = T5ForConditionalGeneration.from_pretrained(model_dir)
        if(args.model_size == '11b'):
            model.parallelize(args.device_map)
        else:
            model.to('cuda:' + str(args.gpu_id))

        tprint('Model loaded in %.1f seconds.' % (time.time() - start_time))

        test_model(dataset, tokenizer, model, base_prompt, args, model_dir)
        del model
    
    return 

if __name__ == '__main__':
  main()
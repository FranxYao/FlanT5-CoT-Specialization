"""Test the BigBench Hard suite"""

import time 
import torch
import re
import json
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
from transformers import T5Tokenizer, T5ForConditionalGeneration
from omegaconf import DictConfig, OmegaConf
from src.utils import tprint, parse_pred_ans

BIGBENCH_PATH = 'BIG-Bench-Hard/bbh/'
BIGBENCH_PROMPT_PATH = 'BIG-Bench-Hard/cot-prompts/'

def load_data(dataset_name):
    data = json.load(open(BIGBENCH_PATH + dataset_name + '.json'))
    return data['examples']

def parse_ans_boolean_expressions(ans, target):
    if('the answer is' in ans):
        ans_ = ans.split('the answer is ')[1]
        if('True' in ans_): ans_ = 'True'
        else: ans_ = 'False'
    else: ans_ = 'NULL'
    # if(len(ans_) > 1):
    #     ans_ = ans_[1][1]
    # else: 
    #     ans_ = 'A'
    return ans_ == target, ans_

PARSE_FN = {'boolean_expressions': parse_ans_boolean_expressions}

def modify_prompt(prompt):
    prompt_q = prompt.split('A:')[0]
    prompt_answer = prompt.split('the answer is ')[1]
    prompt_new = prompt_q + 'A: the answer is ' + prompt_answer
    return prompt_new

def parse_ans_general(ans, target, prompt_mode):
    if(prompt_mode == 'cot'):
        if('the answer is' in ans):
            ans_ = ans.split('the answer is')[1]
            if(target in ans_): return True, ans_
            else: return False, ans_
        else: return False, ans
    else: # prompt_mode == 'ao'
        if(target in ans): return True, ans
        else: return False, ans

def test_model(dataset_name, dataset, tokenizer, model, base_prompt, args, model_dir):
    """Test model on BBH dataset"""
    tprint('Start decoding %s, %d cases... ' % (dataset_name, len(dataset)))
    # parse_ans = PARSE_FN[dataset_name]
    parse_ans = parse_ans_general
    i = 0

    output_path = args.output_path + dataset_name + '_' + args.prompt_mode + '_' + model_dir.split('/')[-1] + '.txt'
    tprint('Model output to: %s' % output_path)

    if(isinstance(args.batch_size, int)):
        batch_size = args.batch_size
    else:
        batch_size = args.batch_size[dataset_name]
    
    acc = 0
    with open(output_path, 'w') as fd:
        tqdm_total = len(dataset) // batch_size
        if(len(dataset) % batch_size != 0): tqdm_total += 1
        for i in tqdm(range(0, len(dataset), batch_size), total=tqdm_total):
            questions = []
            q_batch = []
            a_batch = []
            for k in range(batch_size):
                if(i + k >= len(dataset)): break
                
                q = dataset[i + k]['input'] 
                q_batch.append(q)
                a = dataset[i + k]['target'] 
                a_batch.append(a)
                if(args.prompt_mode == 'cot'):
                    prompt_q = base_prompt + '\n\nQ: ' + q + '\n' + "A: Let's think step by step.\n"
                else: # prompt_mode == 'ao'
                    prompt_q = base_prompt + '\n\nQ: ' + q + '\n' + 'A: the answer is '

                questions.append(prompt_q)
                
            inputs = tokenizer(questions, padding=True, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(inputs['input_ids'].to(model.device), 
                                         attention_mask=inputs['attention_mask'].to(model.device), 
                                         max_length=256
                                         )
            
            for q, a, ans_ in zip(q_batch, a_batch, outputs):
                ans_raw = tokenizer.decode(ans_).replace('<pad>', '').strip()
                acc_, parsed_ans = parse_ans(ans_raw, a, args.prompt_mode)
                fd.write('Q: %s\nA_model:\n%s\n%s\nA:\n%s\n\n' % (q, ans_raw, parsed_ans, a))
                acc += acc_

        fd.write('\n\n----\nEXAMPLE PROMPT: %s\n\n' % prompt_q)
        fd.write('\n\n----\nEXAMPLE OUTPUT: %s\n\n' % ans_raw)

    tprint('%s %d questions %d correct, acc %.4f' % (dataset_name, len(dataset), acc, acc / len(dataset)))
    acc = acc / len(dataset)
    return acc

def load_and_test(model_dir, args, datasets, prompts, tokenizer):
    start_time = time.time()
    tprint('Loading the model from %s' % model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)

    if(args.model_size == '11b'):
        # import ipdb; ipdb.set_trace()
        model.parallelize(args.device_map)
    else:
        model.to('cuda:' + str(args.gpu_id))

    tprint('Model loaded in %.1f seconds.' % (time.time() - start_time))
    all_perf = []

    for dataset_name in datasets:
        base_prompt = prompts[dataset_name]
        if(args.prompt_mode == 'ao'): # else, cot prompt, do nothing
            base_prompt = modify_prompt(base_prompt)

        dataset = datasets[dataset_name]
        acc = test_model(dataset_name, dataset, tokenizer, model, base_prompt, args, model_dir)
        all_perf.append(acc)
        # tprint('%.4f' % acc)
    acc = np.average(all_perf)
    tprint('All average %.4f' % np.average(all_perf))
    return acc

@hydra.main(version_base=None, config_path="src/conf", config_name="config_inference_bbh")
def main(args : DictConfig):
    tprint(OmegaConf.to_yaml(args))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer)

    datasets = {}
    prompts = {}
    for dataset_name in args.batch_size:
        tprint('Loading dataset: %s' % dataset_name)
        datasets[dataset_name] = load_data(dataset_name)
        prompts[dataset_name] = open(BIGBENCH_PROMPT_PATH + dataset_name + '.txt').read()

    if(args.base_model in ['google/flan-t5-large', 'google/flan-t5-xl', 'google/flan-t5-xxl']): # test initial checkpoint
        model_dir = args.base_model
        load_and_test(model_dir, args, datasets, prompts, tokenizer)
    else: # test specialized model
        results = []
        for i in args.iter:
            model_dir = args.base_model + 'iter_' + str(i)
            acc = load_and_test(model_dir, args, datasets, prompts, tokenizer)
            results.append(acc)

        model_dir = args.base_model + 'end'
        acc = load_and_test(model_dir, args, datasets, prompts, tokenizer)
        results.append(acc)
        for acc in results:
            tprint('%.4f' % acc)
    return 

if __name__ == '__main__':
  main()
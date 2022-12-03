"""Data utils for the gsm8k dataset"""

import time
from .utils import tprint

def parse_codex_outputs(lines):
    questions = []
    answers = []
    ans_pred = []
    per_step_prob = []
    mode = ""
    ans_list = None
    for li, l in enumerate(lines):
        if(l.startswith('Question')):
            if(mode == 'm' or mode == 'none'):
                mode = 'none'
                continue
            q = l
            mode = 'q'
            if(ans_list is not None):
                ans_pred.append(ans_list)
                per_step_prob.append(prob_list)
            ans_list = []
            prob_list = []
        elif(l.startswith('Answer: ')):
            questions.append(q)
            a = [l]
            mode = 'a'
        elif(l.startswith('Model output')):
            if(l.startswith('Model output 0')):
                answers.append(''.join(a))
            mode = 'm'
            m = [l]
        elif(l.startswith('Per-step')):
            ans_list.append(''.join(m))
            mode = 'p'
            prob_list.append(l)
        else:
            if(mode == 'a'):
                a.append(l)
            elif(mode == 'm'):
                m.append(l)
            elif(mode == 'p'): 
                pass
            elif(mode == 'none'):
                pass
            else:
                print(mode)
                print(li)
                print(lines[li - 1])
                print(l)
                raise ValueError() 
    ans_pred.append(ans_list)
    per_step_prob.append(prob_list)
    
    per_step_prob_ = []
    for p in per_step_prob:
        p_ = []
        for pi in p: 
            p_.append(pi.split(' ||| '))
        per_step_prob_.append(p_)
    return questions, answers, ans_pred, per_step_prob_

def load_codex_generated(data_path):
    """load the codex generated gsm8k data
    
    Args:
        data_path (str): path to the codex generated gsm8k data

    Returns:

    """
    tprint('Loading codex generated gsm8k data ... ')
    start_time = time.time()
    data_path += 'train_codex_complex_duplicate.txt'

    # load positive data
    positive_lines = open(data_path).readlines()
    questions, answers, ans_pred, per_step_prob_ = parse_codex_outputs(positive_lines)

    # load negative data
    return 

def match_gpt3_token_to_t5_token(gpt3_token_probs):
    """Match the gpt3 token to t5 token

    Args:
    """
    return 
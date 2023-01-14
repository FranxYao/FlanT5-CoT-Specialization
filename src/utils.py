from datetime import datetime
import pytz 
import re
import editdistance
import matplotlib
import torch

import re 
import numpy as np 
import matplotlib.pyplot as plt

from tqdm import tqdm 
from collections import Counter

matplotlib.rcParams['figure.dpi'] = 200

def parse_step_prob_codex(p):
    p = p.replace('Per-step decode:', '')
    if("'<<'" in p): 
        assert("'>>'" not in p)
        p = p.split('>>')
        tok = p[0][3:-1]
        p_ = []
        for pi in p[1:6]:
            prob = float(pi[:8].strip())
            p_.append((tok, prob))
            tok = pi[11:-1]
        prob = float(p[6].strip())
        p_.append((tok, prob))
    else: 
        assert("'<<'" not in p)
        p = p.split('<<')
        p_ = []
        p[-1] += ' '
        for pi in p:
            if(len(pi.strip()) == 0): continue
            pi = [pi[:-10], pi[-8:]]
            tok = pi[0][1:-1].replace('\\n', '\n')
            prob = float(pi[1].strip())
            p_.append((tok, prob))
    return p_


def parse_step_prob_codex_prev(p):
    if("'<<'" in p): print(p)
    assert("'<<'" not in p)
    p = p.split('<<')
    p_ = []
    for pi in p:
        if('>>' not in pi): continue
        if("'>>'" not in pi):
            pi = pi.split('>>')
            assert(len(pi) == 2)
        else: 
            pi = [pi[:-10], pi[-8:]]
            print(pi)
        tok = pi[0][1:-1].replace('\\n', '\n')
        prob = float(pi[1].strip())
                
        p_.append((tok, prob))
    return p_


def parse_codex_outputs(lines):
    """Parse Codex outputs into question, answer, prediction, and per-step-probs

    NOTE: THIS FUNCTION HAS BUG WHEN ADDING QUESTIONS. LATER THE OUTPUT IS CORRECTED MANUALLY.
    BE CAREFUL IF WANT TO USE THIS FUNCTION.
    """
    questions = []
    answers = []
    ans_pred = []
    per_step_prob = []
    mode = ""
    ans_list = None
    for li, l in tqdm(enumerate(lines), total=len(lines)):
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
            questions.append(q) # NOTE: BUG HERE 
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
    for p in tqdm(per_step_prob): # p means one question
        p_ = []
        for i, pi in enumerate(p): # pi means one answer for one question
            if(i == 0): pi = pi.split('Per-step decode: ')[1] 
            pi = pi.split(' ||| ')
            pi_ = []
            ans_end = "false"
            for pij in pi: # pij means one step in one answer
                pij_ = parse_step_prob_codex(pij)
                if(len(pij_) == 0): continue
                
                if('answer' in pij_[0][0]): 
                    ans_end = "ready1"
                if('is' in pij_[0][0]): 
                    if(ans_end == "ready1"):
                        ans_end = "ready2"
                if('Question' in pij_[0][0] and ans_end == "ready2"): 
                    ans_end = "true"
                if(ans_end == "true"): break
                
                pi_.append(pij_)
            p_.append(pi_)
        per_step_prob_.append(p_)
    
    assert(len(questions) == len(answers) == len(ans_pred) == len(per_step_prob_))
    return questions, answers, ans_pred, per_step_prob_


def is_step_break(tok, before, after):
    if(tok == '\n'): return True
    else: 
        if(tok in [',', '.']):
            if(len(re.findall('[0-9]+', before)) == 0 and len(re.findall('[0-9]+', after)) == 0):
                return True
    return False


def vis_prob_flow(questions, answers, ans_pred, per_step_prob, qid, aid):
    print(questions[qid])
    print(answers[qid])
    print(ans_pred[qid][aid])
    
    probs = []
    tokens = []
    step_breaks = []
    j = 0
    for i, tp in enumerate(per_step_prob[qid][aid]):
        if(len(tp) == 0): continue
        if(i < 2 and ('step' in tp[0][0] or '\n' in tp[0][0])): continue
        tok = tp[0][0]
        if(i > 0): before = per_step_prob[qid][aid][i - 1][0][0]
        else: before = ''
        if(i < len(per_step_prob[qid][aid]) - 1): after = per_step_prob[qid][aid][i + 1][0][0]
        else: after = 0 
        if(is_step_break(tok, before, after)): step_breaks.append(j)
        tok = tok.replace("$", "\\$").replace('\n', '\\n')
        tokens.append(tok)
        probs.append(tp[0][1])
        j += 1
    
    cmap_mpl = plt.get_cmap("YlGnBu")
    
    fig, ax = plt.subplots()
    r = 0.3
    fig.set_size_inches(r * len(tokens), 3.5, forward=True)
    ax.bar(np.arange(len(tokens)), np.array(probs), color=cmap_mpl(probs))
    ax.axhline(y=0.5, color='tomato', linestyle="-.")
    for b in step_breaks: ax.axvline(x=b+0.5, color='tomato')
    
    plt.xticks(ticks=np.arange(len(tokens)), labels=tokens, fontsize=10, rotation=60)
    plt.show()
    return 


def vis_heatmap(questions, answers, ans_pred, per_step_prob, qid, aid):
    probs = []
    tokens = []
    for tp in per_step_prob[qid][aid]:
        if(len(tp) == 0): continue
        tp0 = tp[0]
        tp1 = list(tp[1:])
        tp1.sort(key=lambda x:x[1], reverse=True)
        tp_ = [tp0]
        tp_.extend(tp1)

        prob = [p[1] for p in tp_]
        token = [t[0] for t in tp_]
        if(len(prob) != 6):
            print(tp)
            continue
        probs.append(prob)
        tokens.append(token)
    
    print(questions[qid])
    print(answers[qid])
    print(ans_pred[qid][aid])

    probs = np.array(probs)

    T = len(probs)

    fig, ax = plt.subplots()
    r = 0.8
    fig.set_size_inches(r * 6, r * T, forward=True)

    im = ax.imshow(probs[:T] + 0.2, aspect=0.2, cmap="YlGnBu")

    for i in range(T):
        for j in range(6):
            tok = tokens[i][j]
            if(tok == "\n"): tok = "\\n"
            if(tok == "\n\n"): tok = "\\n\\n"
            tok = tok.replace("$", "\\$")
            if(probs[i, j] < 0.5): color="grey"
            else: color="whitesmoke"
            text = ax.text(j, i, 
                            tok + '   '+ str(probs[i, j]),
                            ha="center", va="center", color=color, fontsize=5)
            if(tokens[i][j] == "\n" and j == 0):
                text = ax.text(j - 0.7, i, 
                            ">>",
                            ha="center", va="center", color="dimgray", fontsize=6)
    plt.tick_params(left=False, labelleft = False)
    plt.show()
    return 


def parse_step_prob_flan(p):
    assert("'<<'" not in p)
    p = p.split('<<')
    p_ = []
    for pi in p:
        if(">>" not in pi):
            continue
        pi = [pi[:-10], pi[-8:]]
        tok = pi[0][1:-1].replace('\\n', '\n')
        prob = float(pi[1].strip())
        p_.append((tok, prob))
    return p_


def parse_flan_t5_outputs(lines):
    """Parse Flan-T5 outputs into question, answer, prediction, and per-step-probs
    
    TODO: ignore unstopped answer chunks
    TODO: compare
    """
    questions = []
    answers = []
    ans_pred = []
    per_step_prob = []
    mode = ""
    ans_list = None
    for li, l in tqdm(enumerate(lines), total=len(lines)):
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
            m = [l.replace('<pad>', '').strip()]
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
    for p in tqdm(per_step_prob): # p means one question
        p_ = []
        for i, pi in enumerate(p): # pi means one answer for one question
            if(i == 0): pi = pi.split('Per-step decode: ')[1] 
            pi = pi.split(' ||| ')
            pi_ = []
            
            for pij in pi: # pij means one step in one answer
                pij_ = parse_step_prob_flan(pij)
                if(len(pij_) > 0):
                    if('<pad>' in pij_[0][0]): break
                    pi_.append(pij_)
            p_.append(pi_)
        per_step_prob_.append(p_)
    
    return questions, answers, ans_pred, per_step_prob_


# def test_answer(pred_str, ans_str):
#     pattern = '\d*\.?\d+'
#     pred = re.findall(pattern, pred_str)
#     if(len(pred) >= 1):
#         pred = pred[-1]
#         gold = re.findall(pattern, ans_str)
#         gold = gold[-1]
#         return pred == gold
#     else: return False


def test_answer(pred_str, ans_str):
    """Find the last number as the predicted answer"""
    pattern = '\d*\.?\d+'
    pred = re.findall(pattern, pred_str)
    # print(pred_str)
    # print(ans_str)
    if(len(pred) >= 1):
        pred = float(pred[-1])
        gold = re.findall(pattern, ans_str)
        if(len(gold) == 0): return -1
        gold = float(gold[-1])
        if(pred == gold): return 1
        else: return 0
    else: return 0

def test_acc(ans_pred, answers):
    acc = 0
    for ap, a in zip(ans_pred, answers):
        if(test_answer(ap[0], a)): acc += 1
    print('total %d, pred %d, acc %.4f' % (len(ans_pred), acc, acc / len(ans_pred)))
    return acc / len(ans_pred)


def find_answer(pred_str):
    pattern = '\d*\.?\d+'
    pred = re.findall(pattern, pred_str)
    if(len(pred) >= 1): pred = pred[-1]
    else: pred = None
    return pred


def find_majority(ans_strs):
    ans = []
    for s in ans_strs:
        pred = find_answer(s)
        ans.append(pred)
    ans = Counter(ans)
    major = ans.most_common(1)[0][0]
    return major


def majority_vote_acc(ans_pred, answers):
    acc = 0
    ans_labels = []
    for ap, a in zip(ans_pred, answers):
        major = find_majority(ap)
        if(test_answer(major, a)):
            acc += 1
        labels = []
        for api in ap:
            if(test_answer(api, a)): labels.append(1)
            else: labels.append(0)
        ans_labels.append(labels)
    print('total %d, pred %d, acc %.4f' % (len(ans_pred), acc, acc / len(ans_pred)))
    return acc, ans_labels


# def parse_pred_ans_multiarith(filename, stop_at=-1):
#     with open(filename) as fd: lines = fd.readlines()
#     am, a = None, None
#     num_q, acc = 0, 0
#     current_mode = 'none'
#     questions = []
#     ans_pred = []
#     ans_gold = []
#     for l in lines:
#         if(l.startswith('Q: ')):
#             if(am is not None and a is not None):
#                 questions.append(q)
#                 ans_pred.append(am)
#                 ans_gold.append(a)
#                 if(test_answer(am, a)):
#                     acc += 1
#             current_mode = 'q'
#             q = l
#             num_q += 1
#             if(num_q == stop_at): break
#         elif(l.startswith('A_model:')):
#             current_mode = 'am'
#             am = l
#         elif(l.startswith('A:')):
#             current_mode = 'a'
#             a = l
#         else:
#             if(current_mode == 'q'): q += l
#             elif(current_mode == 'am'): am += l
#             elif(current_mode == 'a'): a += l
#             else:
#                 raise ValueError(current_mode)
                
#     questions.append(q)
#     ans_pred.append(am)
#     ans_gold.append(a)
#     if(test_answer(am, a)):
#         acc += 1
#     print('num_q %d correct %d ratio %.4f' % (num_q, acc, float(acc / num_q)))
#     return questions, ans_pred, ans_gold

def parse_pred_ans(filename):
    with open(filename) as fd: lines = fd.readlines()
    am, a = None, None
    num_q, acc, skipped = 0, 0, 0
    current_mode = 'none'
    questions = []
    ans_pred = []
    ans_gold = []
    for l in lines:
        if(l.startswith('Q: ')):
            if(am is not None and a is not None):
                questions.append(q)
                ans_pred.append(am)
                ans_gold.append(a)
                test_result = test_answer(am, a)
                if(test_result == 1):
                    acc += 1
                elif(test_result == -1):
                    skipped += 1
            current_mode = 'q'
            q = l
            num_q += 1
        elif(l.startswith('A_model:')):
            current_mode = 'am'
            am = l
        elif(l.startswith('A:')):
            current_mode = 'a'
            a = l
        else:
            if(current_mode == 'q'): q += l
            elif(current_mode == 'am'): am += l
            elif(current_mode == 'a'): a += l
            else:
                raise ValueError(current_mode)
                
    questions.append(q)
    ans_pred.append(am)
    ans_gold.append(a)
    test_result = test_answer(am, a)
    if(test_result == 1):
        acc += 1
    elif(test_result == -1):
        skipped += 1
    print('num_q %d correct %d ratio %.4f skipped %d' % (num_q, acc, float(acc / num_q), skipped))
    acc = float(acc / num_q)
    return questions, ans_pred, ans_gold, acc


def tprint(str):
    timezone = pytz.timezone("America/Vancouver")
    timenow = datetime.now(timezone)
    currenttime= timenow.strftime("%m/%d/%Y, %H:%M:%S")
    print(currenttime + ' ' + str)
    return


def print_transformed_probs(tokenizer, transformed):
    s = list(t[0][0] for t in transformed)
    s = tokenizer.decode(tokenizer.convert_tokens_to_ids(s))
    print(s)
    return 


class ClosestToken(object):

    def __init__(self, vocab):
        self.vocab = vocab
        self.closest_dict = {'\n': '▁'}
        return
    
    def __call__(self,  token):
        if(token in self.closest_dict): 
            return self.closest_dict[token]
        else: 
            if(token in self.vocab): 
                self.closest_dict[token] = token
                return token
            else:
                edit_dist = 100000
                t_close = None
                for t in self.vocab:
                    d = editdistance.eval(t, token)
                    if(d < edit_dist):
                        edit_dist = d
                        t_close = t
                self.closest_dict[token] = t_close
            return t_close


def contains_char_or_number(s):
    # Use the 'search' method of the re module to check if the string
    # contains either a character (a-z or A-Z) or a number (0-9)
    # Produced by ChatGPT
    return re.search(r'[a-zA-Z0-9]', s) is not None


def transform_codex_token_to_t5_token_by_tokenizer_alignment(
    qid, aid, codex_per_step_probs, tokenizer):
    """Transform codex token to t5 token and take care of some details

    Advantage: converted stenctence is exactly the same as the original sentence
    Disadvantage: there may exist complex k-to-n mapping between codex token and t5 token

    Algorithm: use flan tokenizer to tokenize the codex sentence, then use DTW to obtain the alignment
    """
    # TBC 
    return 


def transform_codex_token_to_t5_token(qid, aid, codex_per_step_probs, closest_token):
    """Transform codex token to t5 token and take care of some details

    Algorithm: for each codex token, find the closest t5 token

    Advantage: this is a one-to-one mapping
    Disadvantage: some mapping may be wrong. converte sentence may not be exactly the same 
    as the original sentence

    TODO: use flan tokenizer to tokenize the codex sentence, then use DTW to obtain the alignment
    TODO: Mark if a token is a transition step then put different loss weights on it
    """
    transform_result = {'blank_before_number': 0, 
                        "blank_after_number": 0, 
                        'blank_step_before_number': 0,
                        'blank_step_after_number': 0
                        }
    loss_mask = []
    blank_step = []
    for _ in range(6):
        blank_step.append(('▁', 0))
    end_step = [('</s>', 1.), ('</s>', 1.), ('▁', 0), ('▁', 0), ('▁', 0), ('▁', 0)]
    transferred_per_step_probs = []
    t_prev = None
    assert(codex_per_step_probs[0][0][0] == ' step')

    start_idx = 1
    while(contains_char_or_number(codex_per_step_probs[start_idx][0][0])):
        start_idx += 1
    if(start_idx >= 3): 
        print('questions %d, answer %d, start_idx %d may need double check' % (qid, aid, start_idx))

    end_idx = len(codex_per_step_probs)
    if(codex_per_step_probs[-1][0][0] == '\n'): 
        end_idx = end_idx - 1
        if(codex_per_step_probs[2][0][0] == '\n'): end_idx = end_idx - 1
    
    for codex_per_step in codex_per_step_probs[start_idx :end_idx]:
        transferred_per_step = []
        for tpi, tp in enumerate(codex_per_step):
            t, p = tp
            t_close = closest_token(t)
            if(tpi == 0):
                # if(t_prev is not None and '.' in t_prev):
                #    print(t, t_close, t_prev)
                #    import ipdb; ipdb.set_trace()
                if(t_prev is not None and 
                    t_close[0].isdigit() and 
                    not t_prev[-1].isdigit() and 
                    t_prev[-1] != '.'
                    ):
                    if("▁" + t_close in closest_token.vocab):
                        t_close = "▁" + t_close # TODO: add the underscore to the corresponding token in the top 5 list
                        transform_result['blank_before_number'] += 1
                    else: 
                        transferred_per_step_probs.append(blank_step)
                        loss_mask.append(0)
                        transform_result['blank_step_before_number'] += 1
                if(t_prev is not None and 
                    t_close[0] != '▁' and 
                    not t_close[0].isdigit() and 
                    t_prev[-1].isdigit() and 
                    t_close[0] != '.'
                    ):
                    if("▁" + t_close in closest_token.vocab):
                        t_close = "▁" + t_close # TODO: add the underscore to the corresponding token in the top 5 list
                        transform_result['blank_after_number'] += 1
                    else: 
                        transferred_per_step_probs.append(blank_step)
                        loss_mask.append(0)
                        transform_result['blank_step_after_number'] += 1
                t_prev = t_close
            transferred_per_step.append((t_close, p))
        transferred_per_step_probs.append(transferred_per_step)
        loss_mask.append(1)
    transferred_per_step_probs.append(end_step)
    return transferred_per_step_probs, loss_mask, transform_result


def kl_divergence(p0, p1, eps=1e-10):
    """Calculate the kl divergence between two distributions
    Args: 
        p0: size=[*, support_size]
        p1: size=[*, support_size]
    """
    kld = p0 * torch.log(p0 / (p1 + eps) + eps)
    kld = kld.sum(dim=-1)
    return kld


def get_optimizer(optimizer_name, model):
    """Get optimizer, currently supporting AdamW and AdaFactor
    """
    # TBC
    return 

def dtw(series_1, series_2, norm_func = np.linalg.norm):
    """Use dynamic time wrapping to align to tokenizers, modified from:
    
    https://github.com/talcs/simpledtw/blob/master/simpledtw.py
    """
    matrix = np.zeros((len(series_1) + 1, len(series_2) + 1))
    matrix[0,:] = np.inf
    matrix[:,0] = np.inf
    matrix[0,0] = 0
    for i, vec1 in enumerate(series_1):
        for j, vec2 in enumerate(series_2):
            cost = norm_func(vec1, vec2)
            matrix[i + 1, j + 1] = cost + min(matrix[i, j + 1], matrix[i + 1, j], matrix[i, j])
    matrix = matrix[1:,1:]
    i = matrix.shape[0] - 1
    j = matrix.shape[1] - 1
    matches = []
    mappings_series_1 = [list() for v in range(matrix.shape[0])]
    mappings_series_2 = [list() for v in range(matrix.shape[1])]
    while i > 0 or j > 0:
        matches.append((i, j))
        mappings_series_1[i].append(j)
        mappings_series_2[j].append(i)
        option_diag = matrix[i - 1, j - 1] if i > 0 and j > 0 else np.inf
        option_up = matrix[i - 1, j] if i > 0 else np.inf
        option_left = matrix[i, j - 1] if j > 0 else np.inf
        move = np.argmin([option_diag, option_up, option_left])
        if move == 0:
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    matches.append((0, 0))
    mappings_series_1[0].append(0)
    mappings_series_2[0].append(0)
    matches.reverse()
    for mp in mappings_series_1:
        mp.reverse()
    for mp in mappings_series_2:
        mp.reverse()

    return matches, matrix[-1, -1], mappings_series_1, mappings_series_2, matrix
"""Utility functions used by notebooks"""

import re
import matplotlib

import numpy as np 
import matplotlib.pyplot as plt

from tqdm import tqdm 
from collections import Counter

matplotlib.rcParams['figure.dpi'] = 200

def parse_step_prob_codex(p):
    p = p.replace('Per-step decode:', '')
    # print(p)
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
        # ans_end = 'false'
        # print(p)
        p[-1] += ' '
        for pi in p:
            if(len(pi.strip()) == 0): continue
            pi = [pi[:-10], pi[-8:]]
            # print(pi)
            tok = pi[0][1:-1].replace('\\n', '\n')
            prob = float(pi[1].strip())
            p_.append((tok, prob))
    return p_

def parse_step_prob_codex_prev(p):
    if("'<<'" in p): print(p)
    assert("'<<'" not in p)
    p = p.split('<<')
    p_ = []
    # ans_end = 'false'
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
        
        # if('answer' in tok): ans_end == "ready1"
        # if('is' in tok and ans_end == "ready1"): ans_end = "ready2"
        # if('Question' in tok and ans_end == "ready2"): ans_end = "true"
        # if(ans_end == "true"): break
        # print(tok, ans_end)
                
        p_.append((tok, prob))
    return p_

def parse_codex_outputs(lines):
    """Parse Codex outputs into question, answer, prediction, and per-step-probs
    
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
                # print(pij_)
                if(len(pij_) == 0): continue
                
                if('answer' in pij_[0][0]): 
                    ans_end = "ready1"
                    # print(pij_[0][0], ans_end)
                if('is' in pij_[0][0]): 
                    # print(pij_[0][0], ans_end)
                    if(ans_end == "ready1"):
                        ans_end = "ready2"
                if('Question' in pij_[0][0] and ans_end == "ready2"): 
                    ans_end = "true"
                if(ans_end == "true"): break
                
                # print(pij_[0][0], ans_end, 'answer' in pij_[0][0])
                pi_.append(pij_)
            p_.append(pi_)
        per_step_prob_.append(p_)
    
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
        # TODO: rank prob
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
    # TODO: resize figure according to sentence length

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
                            # tokens[i][j],
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
    # ans_end = 'false'
    for pi in p:
        # print(p)
        if(">>" not in pi):
            # print('!!')
            # print(pi)
            continue
        # if('>>' not in pi): continue
        # if("'>>'" not in pi):
        #     pi = pi.split('>>')
        #     if(len(pi) != 2):
        #         print(p)
        #     assert(len(pi) == 2)
        # else: 
        # print(pi)
        pi = [pi[:-10], pi[-8:]]
        # print(pi)
        tok = pi[0][1:-1].replace('\\n', '\n')
        prob = float(pi[1].strip())
        
        # if('answer' in tok): ans_end == "ready1"
        # if('is' in tok and ans_end == "ready1"): ans_end = "ready2"
        # if('Question' in tok and ans_end == "ready2"): ans_end = "true"
        # if(ans_end == "true"): break
        # print(tok, ans_end)
                
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

def test_answer(pred_str, ans_str):
    pattern = '\d*\.?\d+'
    pred = re.findall(pattern, pred_str)
    if(len(pred) >= 1):
        # print(pred_str)
        pred = pred[-1]
        gold = re.findall(pattern, ans_str)
        # print(ans_str)
        gold = gold[-1]
        return pred == gold
    else: return False

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
    # print(ans)
    major = ans.most_common(1)[0][0]
    return major

def majority_vote_acc(ans_pred, answers):
    acc = 0
    for ap, a in zip(ans_pred, answers):
        major = find_majority(ap)
        if(test_answer(major, a)):
            acc += 1
    print('total %d, pred %d, acc %.4f' % (len(ans_pred), acc, acc / len(ans_pred)))
    return acc
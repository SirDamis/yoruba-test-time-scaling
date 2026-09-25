import argparse
import json
import re
import jsonlines
from fraction import Fraction
# from vllm import LLM, SamplingParams
import sys
import os
import pdb
import util
import grader
from transformers import AutoTokenizer, AutoModelForCausalLM
from numpy import *
import torch
from peft import PeftModel,PeftConfig

MAX_INT = sys.maxsize
INVALID_ANS = "[invalid]"



def extract_answer(backbone,completion,test_name):
    if test_name == 'math500':
        return extract_answer_math500(backbone,completion)
    elif test_name == 'mgsm':
        return extract_answer_mgsm(backbone,completion)

def extract_answer_mgsm(backbone,completion):
    if 'mathoctopusparallel' in backbone or 'Llama'in backbone:
        text = re.sub(r"(\d),(\d)", "\g<1>\g<2>", completion)  
        res = re.findall(r"(\d+(\.\d+)?)", text)  
        if len(res) > 0:
            num_str = res[-1][0]
            return float(num_str)
        else:
            return None
    elif 'Qwen' in backbone or 'deepseek' in backbone:
        matches = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', completion)
        if len(matches)>0:
            text = re.sub(r"(\d),(\d)", "\g<1>\g<2>", matches[0])  
            res = re.findall(r"(\d+(\.\d+)?)", text)  
            if len(res) > 0:
                num_str = res[-1][0]
                return float(num_str)
            else:
                return None
        else:
            return None
    else:
        text = completion.split('The answer is: ')
        if len(text) > 1:
            extract_ans = text[-1].strip()
            match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
            if match:
                if '/' in match.group():
                    denominator = match.group().split('/')[1]
                    numerator = match.group().split('/')[0]
                    if is_number(denominator) == True and is_number(numerator) == True:
                        if denominator == '0':
                            return round(float(numerator.replace(',', '')))
                        else:
                            frac = Fraction(match.group().replace(',', ''))
                            num_numerator = frac.numerator
                            num_denominator = frac.denominator
                            return round(float(num_numerator / num_denominator))
                    else:
                        return None
                else:
                    if float(match.group().replace(',', '')) == float('inf'):
                        return None
                    return round(float(match.group().replace(',', '')))
            else:
                return None
        else:
            return None

def extract_answer_math500(backbone,completion):
    if 'mathoctopusparallel' in backbone.lower() or 'llama3.2' in backbone.lower():
        split_ans = completion.strip().replace('</s>','').replace('<|end_of_text|>','').replace('<|im_end|>','').strip().replace('<|eot_id|>','').split('####')
    elif 'qwen' in backbone.lower() or 'deepseek' in backbone.lower():
        matches = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', completion)
        if len(matches)>0:
            return matches[0]
        else:
            return INVALID_ANS
    else:
        split_ans = completion.strip().replace('</s>','').replace('<|end_of_text|>','').replace('<|im_end|>','').strip().replace('<|eot_id|>','').split('The answer is:')
    if len(split_ans) > 1:
        ans = split_ans[-1].strip()
        return ans
    else:
        return INVALID_ANS

def judge_answer(y_pred, prompt_answer,testset_name):
    
    if testset_name == 'math500':
        # print(y_pred, prompt_answer,util.is_equiv(y_pred, prompt_answer))
        return grader.grade_answer(y_pred, prompt_answer)
    elif testset_name == 'mgsm':
        # print(y_pred, prompt_answer,util.is_equiv(y_pred, prompt_answer))
        return util.is_equiv(y_pred, prompt_answer)
        
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False
        
def find_most_frequent_answers(max_seed,answer_list):
    cnt = len(answer_list[0])
    assert len(answer_list) == max_seed and all(len(sublist) == cnt for sublist in answer_list)
  
    final_answers = [None] * cnt
    
    for question_idx in range(cnt):
        answer_frequency = {}
        
        for answer_idx in range(max_seed):
            answer = answer_list[answer_idx][question_idx]
            if answer != "[invalid]":
                if answer in answer_frequency:
                    answer_frequency[answer] += 1
                else:
                    answer_frequency[answer] = 1

        if not answer_frequency:
            final_answers[question_idx] = "[invalid]"
        else:
            final_answers[question_idx] = max(answer_frequency, key=answer_frequency.get)
    
    return final_answers

def find_most_frequent_answers_with_reward(max_seed,answer_list, total_reward):
    cnt = len(answer_list[0])
    assert len(answer_list) == max_seed and all(len(sublist) == cnt for sublist in answer_list)
  
    final_answers = [None] * cnt
    
    for question_idx in range(cnt):
        answer_frequency = {}
        answer_accu_reward = {}
        
        for answer_idx in range(max_seed):
            answer = answer_list[answer_idx][question_idx]
            if answer != "[invalid]":
                if answer in answer_accu_reward:
                    answer_accu_reward[answer] += total_reward[answer_idx][question_idx]
                else:
                    answer_accu_reward[answer] = total_reward[answer_idx][question_idx]
        
        if not answer_accu_reward:
            final_answers[question_idx] = "[invalid]"
        else:
            final_answers[question_idx] = max(answer_accu_reward, key=answer_accu_reward.get)
    
    return final_answers

def outcome_reward_model_result(max_seed,sft_result_path_list, MATH_answers,testset_name,prm_score_folder):
    total_ans = []
    total_reward = {}
    ans_len = len(MATH_answers)
    for file_path_idx in range(len(sft_result_path_list)):
        file_path = sft_result_path_list[file_path_idx]

        ans = []
        file_response = []
        with open(file_path) as f:
            for line_idx, line in enumerate(f.readlines()[:ans_len]):
                n = extract_answer(eval(line)[0][1],testset_name)
                file_response.append(eval(line)[0][1])
                reward = eval(line)[1][0]
                if file_path_idx in total_reward.keys():
                    total_reward[file_path_idx].append(reward)
                else:
                    total_reward[file_path_idx] = [reward]
                ans.append(INVALID_ANS if n == None else n)
        assert len(ans) == len(MATH_answers)
        assert len(file_response) == len(ans)
        assert len(total_reward[file_path_idx]) == len(ans)
        total_ans.append(ans) 

    cnt = len(MATH_answers)
    final_ans = [''] * cnt

    max_scores = [-float('inf')] * cnt
    idx_j = ['x'] * cnt
    invalid_response = []
    for i in range(cnt):
        for j in range(max_seed):
            if total_reward[j][i] > max_scores[i] and total_ans[j][i] != INVALID_ANS:
                max_scores[i] = total_reward[j][i]
                final_ans[i] = total_ans[j][i]
                idx_j[i] = j

    result = []
    invalid_outputs = []
    invalid_idx = 0
    for idx, (y_pred, prompt_answer) in enumerate(zip(final_ans, MATH_answers)):
        if y_pred != None and y_pred != INVALID_ANS:
            result.append(judge_answer(y_pred, prompt_answer,testset_name))
        else:
            result.append(False)
            temp = {'output': y_pred, 'answer': prompt_answer}
            invalid_outputs.append(temp)
    acc = sum(result) / len(result)
    # print('len invalid outputs ====', len(invalid_outputs), ', valid_outputs===', invalid_outputs)
    # print('MATH length====', len(result), ', MATH acc====', acc)

    # print(invalid_idx)
    return acc

def process_reward_model_result(max_seed, sft_result_path_list, MATH_answers,MATH_questions,select,model_path,model_name,backbone,lang,adapter_path,testset_name,prm_score_folder):
    score_folder = f'{prm_score_folder}/{backbone}/{testset_name}/{lang}/{model_name}'
    print("score_folder+++++++++++++++++++++++++++++++++++++++",score_folder)
    if not os.path.exists(score_folder):
        os.makedirs(score_folder) 
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to('cuda')

    if adapter_path:
        adapter_config = PeftConfig.from_pretrained(adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)
    if 'mistral' in model_name.lower():
        good_token = '+'
        bad_token = '-'
        step_tag = 'ки'
        candidate_tokens = tokenizer.encode(f"{good_token} {bad_token}") # [648, 387]
    else:
        step_tag = ' \n\n\n\n\n' 
        good_token = ' +'
        bad_token = ' -'
        candidate_tokens = tokenizer.encode(f"{good_token}{bad_token}")
    step_tag_id = tokenizer.encode(f"{step_tag}")[-1] 
    if len(candidate_tokens) > 2:
        candidate_tokens = candidate_tokens[1:]
    assert len(candidate_tokens) == 2 
    # print("candidate_tokens ",candidate_tokens, "step_tag_id ",step_tag_id)
    
    total_ans = []
    total_reward = {}
    for file_path_idx in range(len(sft_result_path_list)):
        file_path = sft_result_path_list[file_path_idx]
        file_name = file_path.split('/')[-1]
        ans = []
        file_reward_res = []
        with open(file_path) as f:
            lines = json.load(f)

        if os.path.exists(f"{score_folder}/{file_name}.score.json"):
            with open(f"{score_folder}/{file_name}.score.json") as g:
                rw_lines = json.load(g)
            for i in range(len(rw_lines)):
                rw = rw_lines[i]
                score_last, score_max, score_mean = rw['last'], rw['max'], rw['mean']
                if select == 'last':
                    reward = score_last
                elif select == 'max':
                    reward = score_max
                elif select == 'mean':
                    reward = score_mean
    
                if file_path_idx in total_reward.keys():
                    total_reward[file_path_idx].append(reward)
                else:
                    total_reward[file_path_idx] = [reward]
                n = extract_answer(backbone,lines[i]['generation'],testset_name)
                ans.append(INVALID_ANS if n == None else n)
            assert len(ans) == len(MATH_answers)
            assert len(total_reward[file_path_idx]) == len(ans)
            total_ans.append(ans) 
        else:       
            for i in range(len(lines)):
                line = lines[i]
                question = MATH_questions[i]
                if 'Qwen' in backbone:
                    line = line['generation'].strip().strip('<|im_end|>').replace('**\n\n',' ').split('\n\n')
                elif 'deepseek' in backbone:
                    line = line['generation'].strip().replace('\n\n','\n').split('\n')
                else:
                    line = line['generation'].strip().strip('</s>').split('\n')
                # line = line['generation'].strip().strip('</s>').split('\n')
                output = []
                if 'Qwen' in backbone:
                    line = line[1:]
                for l in line:
                    if len(l.strip()) > 1:
                        if 'Qwen' in backbone:
                            l = l.replace('**','').replace('\\times','*').replace('\\(','(').replace('\\)',')').replace('\n',' ').strip()
                            text = re.sub(r'\\text\{(.*?)\}', r'\1', l)
                            output.append(text)
                        else:
                            output.append(l.replace('\n','').strip())
                if 'mistral' in model_name.lower():
                    output = f' {step_tag}\n'.join(output)
                    output += step_tag
                else:
                    output = f'{step_tag} '.join(output)
                    output += step_tag
                
                input_for_prm = f"{question} {output}"
                
                input_id = torch.tensor([tokenizer.encode(input_for_prm)]).to('cuda')
    
                with torch.no_grad():
                    logits = model(input_id).logits[:,:,candidate_tokens]
                    scores = logits.softmax(dim=-1)[:,:,0] 
                    step_scores = scores[input_id == step_tag_id]
                # print(step_scores)
                if len(step_scores) == 0:
                    reward_temp = {'last':0,'max':0,'mean':0}
                else:
                    score_last, score_max, score_mean = step_scores[-1], max(step_scores), torch.mean(step_scores)
                    reward_temp = {'last':score_last.cpu().numpy().tolist(),'max':score_max.cpu().numpy().tolist(),'mean':score_mean.cpu().numpy().tolist()}
                file_reward_res.append(reward_temp)
                
                if select == 'last':
                    reward = score_last
                elif select == 'max':
                    reward = score_max
                elif select == 'mean':
                    reward = score_mean
    
                if file_path_idx in total_reward.keys():
                    total_reward[file_path_idx].append(reward)
                else:
                    total_reward[file_path_idx] = [reward]
                n = extract_answer(backbone,lines[i]['generation'],testset_name)
                ans.append(INVALID_ANS if n == None else n)
            assert len(ans) == len(MATH_answers)
            assert len(total_reward[file_path_idx]) == len(ans)
            total_ans.append(ans) 
            
            
            if not os.path.exists(score_folder):
                os.mkdir(score_folder)
            out_f = open(f'{score_folder}/{file_name}.score.json','w')
            json.dump(file_reward_res, out_f,indent=4, ensure_ascii=False)
            out_f.close()

    position = len(MATH_answers)
    final_ans = [''] * position

    max_scores = [-float('inf')] * position
    idx_j = ['x'] * position
    invalid_response = []
    for i in range(position):
        for j in range(max_seed):
            if total_reward[j][i] > max_scores[i] and total_ans[j][i] != INVALID_ANS:
                max_scores[i] = total_reward[j][i]
                final_ans[i] = total_ans[j][i]
                idx_j[i] = j

    result = []
    invalid_outputs = []
    invalid_idx = 0
    for idx, (y_pred, prompt_answer) in enumerate(zip(final_ans, MATH_answers)):
        if y_pred != None and y_pred != INVALID_ANS:
            result.append(judge_answer(y_pred, prompt_answer,testset_name))
        else:
            result.append(False)
            temp = {'output': y_pred, 'answer': prompt_answer}
            invalid_outputs.append(temp)
    acc = sum(result) / len(result)
    # print('len invalid outputs ====', len(invalid_outputs), ', valid_outputs===', invalid_outputs)
    # print('MATH length====', len(result), ', MATH acc====', acc)

    # print(invalid_idx)
    return acc



def self_consistency_and_reward(max_seed,sft_result_path_list, MATH_answers,select,model_name,backbone,lang,testset_name,prm_score_folder):
    score_folder = f'{prm_score_folder}/{backbone}/{testset_name}/{lang}/{model_name}'
        
    total_ans = []
    total_reward = {}
    for file_path_idx in range(len(sft_result_path_list)):
        file_path = sft_result_path_list[file_path_idx]
        file_name = file_path.split('/')[-1]
        ans = []
        with open(file_path) as f:
            lines = json.load(f)

        if os.path.exists(f"{score_folder}/{file_name}.score.json"):
            with open(f"{score_folder}/{file_name}.score.json") as g:
                rw_lines = json.load(g)
            for i in range(len(rw_lines)):
                rw = rw_lines[i]
                score_last, score_max, score_mean = rw['last'], rw['max'], rw['mean']
                if select == 'last':
                    reward = score_last
                elif select == 'max':
                    reward = score_max
                elif select == 'mean':
                    reward = score_mean
    
                if file_path_idx in total_reward.keys():
                    total_reward[file_path_idx].append(reward)
                else:
                    total_reward[file_path_idx] = [reward]
                n = extract_answer(backbone,lines[i]['generation'],testset_name)
                ans.append(INVALID_ANS if n == None else n)
            assert len(ans) == len(MATH_answers)
            assert len(total_reward[file_path_idx]) == len(ans)
            total_ans.append(ans) 
    

    final_ans = find_most_frequent_answers_with_reward(max_seed,total_ans, total_reward)

    result = []
    invalid_outputs = []
    for idx, (y_pred, prompt_answer) in enumerate(zip(final_ans, MATH_answers)):
        if y_pred != None and y_pred != INVALID_ANS:
            result.append(judge_answer(y_pred, prompt_answer,testset_name))
        else:
            result.append(False)
            temp = {'output': y_pred, 'answer': prompt_answer}
            invalid_outputs.append(temp)
    acc = sum(result) / len(result)
    # print('len invalid outputs ====', len(invalid_outputs), ', valid_outputs===', invalid_outputs)
    # print('MATH length====', len(result), ', MATH acc====', acc)
    return acc


def self_consistency(max_seed,sft_result_path_list, MATH_answers,backbone,testset_name):
    
    
        
    total_ans = []
    ans_len = len(MATH_answers)
    for file_path_idx in range(len(sft_result_path_list)):
        file_path = sft_result_path_list[file_path_idx]

        ans = []
        with open(file_path) as f:
            lines = json.load(f)
        for line in lines[:ans_len]:
            n = extract_answer(backbone,line['generation'],testset_name)
            ans.append(INVALID_ANS if n == None else n)
        assert len(ans) == len(MATH_answers)
        total_ans.append(ans)
    final_ans = find_most_frequent_answers(max_seed,total_ans)

    result = []
    invalid_outputs = []
    for idx, (y_pred, prompt_answer) in enumerate(zip(final_ans, MATH_answers)):
        if y_pred != None and y_pred != INVALID_ANS:
            result.append(judge_answer(y_pred, prompt_answer,testset_name))
        else:
            result.append(False)
            temp = {'output': y_pred, 'answer': prompt_answer}
            invalid_outputs.append(temp)
    acc = sum(result) / len(result)
    # print('len invalid outputs ====', len(invalid_outputs), ', invalid_outputs===', invalid_outputs)
    # print('MATH length====', len(result), ', MATH acc====', acc)
    return acc

def MATH_test(data_path, sft_result_dir, max_seed, mode,model_path,model_name,lang,adapter_path,testset_name,res_folder,prm_score_folder):
    folder = sft_result_dir
    print(mode,'+++++++++++++++++++++++++++++++++',folder)
    temp = folder.split('/')
    backbone = temp[-2]
    if not os.path.exists(f'{res_folder}/{backbone}/'):
        os.makedirs(f'{res_folder}/{backbone}/') 
    out_f = open(f'{res_folder}/{backbone}/scores-{testset_name}-{mode}-{lang}-{model_name}.txt','a')
    MATH_answers = []
    MATH_ins = []
    with open(data_path,"r+", encoding="utf8") as f:
        lines = json.load(f)
    for _, item in enumerate(lines):
        temp_instr = item["instruction"]
        MATH_ins.append(temp_instr)
        temp_ans = item['answer']
        if testset_name == 'mgsm':
            temp_ans = int(temp_ans.replace(',', ''))
        MATH_answers.append(temp_ans)

    print('lenght ====', len(MATH_answers))

    sft_result_path_list = [f'{folder}/raw_generation_0.7_{idx}.json' for idx in range(1,max_seed+1,1)]
    sft_result_path_list = [f for f in sft_result_path_list if os.path.exists(f)]
    print(f'Generate seed count: {len(sft_result_path_list)}')

    if mode == 'self-consistency':
        acc = self_consistency(max_seed,sft_result_path_list, MATH_answers,backbone,testset_name)
        print("self-consistency------",acc)
        out_f.write(f'{testset_name} ||| {mode} ||| {max_seed} ||| -- ||| -- ||| {acc}\n')
        out_f.flush()
    elif mode == 'ensemble':
        acc = self_consistency_and_reward(max_seed,sft_result_path_list, MATH_answers,'last',model_name,backbone,lang,testset_name,prm_score_folder)
        print("ensemble-last------",acc)
        out_f.write(f'{testset_name} ||| {mode} ||| {max_seed} ||| {model_name} ||| last ||| {acc}\n')
        out_f.flush()
        acc = self_consistency_and_reward(max_seed,sft_result_path_list, MATH_answers,'max',model_name,backbone,lang,testset_name,prm_score_folder)
        print("ensemble-max------",acc)
        out_f.write(f'{testset_name} ||| {mode} ||| {max_seed} ||| {model_name} ||| max ||| {acc}\n')
        out_f.flush()
        acc = self_consistency_and_reward(max_seed,sft_result_path_list, MATH_answers,'mean',model_name,backbone,lang,testset_name,prm_score_folder)
        print("ensemble-mean------",acc)
        out_f.write(f'{testset_name} ||| {mode} ||| {max_seed} ||| {model_name} ||| mean ||| {acc}\n')
        out_f.flush()
    elif mode == 'prm':
        acc = process_reward_model_result(max_seed,sft_result_path_list, MATH_answers,MATH_ins,'last',model_path,model_name,backbone,lang,adapter_path,testset_name,prm_score_folder)
        print("prm-last------",acc)
        out_f.write(f'{testset_name} ||| {mode} ||| {max_seed} ||| {model_name} ||| last ||| {acc}\n')
        out_f.flush()
        acc = process_reward_model_result(max_seed,sft_result_path_list, MATH_answers,MATH_ins,'max',model_path,model_name,backbone,lang,adapter_path,testset_name,prm_score_folder)
        print("prm-max------",acc)
        out_f.write(f'{testset_name} ||| {mode} ||| {max_seed} ||| {model_name} ||| max ||| {acc}\n')
        out_f.flush()
        acc = process_reward_model_result(max_seed,sft_result_path_list, MATH_answers,MATH_ins,'mean',model_path,model_name,backbone,lang,adapter_path,testset_name,prm_score_folder)
        print("prm-mean------",acc)
        out_f.write(f'{testset_name} ||| {mode} ||| {max_seed} ||| {model_name} ||| mean ||| {acc}\n')
        out_f.flush()
    else:
        acc = outcome_reward_model_result(max_seed,sft_result_path_list, MATH_answers,testset_name,prm_score_folder)
        out_f.write(f'{testset_name} ||| {mode} ||| {max_seed} ||| {model_name} ||| -- ||| {acc}\n')
        out_f.flush()
    
    out_f.close()
    return acc

def main(args):
    
    MATH_test(args.reference_data, args.sft_output_files, args.seed, args.mode,args.rm_model,args.model_name,args.lang,args.adapter_path,args.testset_name,args.res_folder,args.prm_score_folder)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--rm_model", default="", type=str, help="model path")
    parser.add_argument("--reference_data", default="", type=str, help="reference test data")
    parser.add_argument("--testset_name", default="", type=str, help="math500,mgsm")
    parser.add_argument("--model_name", default="", type=str, help="prm model_name")
    parser.add_argument("--lang", type=str, default="", help="langauge")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    parser.add_argument("--mode", type=str, default="", help="self-consistency,ensemble,prm")
    parser.add_argument("--adapter_path", type=str, default="", help="adapter_path")    
    parser.add_argument("--sft_output_files", default="", type=str, help="multiple output results from different seeds")
    parser.add_argument("--prm_score_folder", default="", type=str, help="store scores verified by prm")
    parser.add_argument("--res_folder", default="", type=str, help="store prm verifier results")
    args = parser.parse_args()

    main(args)



from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
from datasets import load_dataset
import argparse
import os
from peft import PeftModel
from peft import get_peft_model, LoraConfig, TaskType
# Ensure bitsandbytes is available for 8-bit quantization
# import bitsandbytes as bnb
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

from torch.nn import BCEWithLogitsLoss
from transformers import DataCollatorWithPadding
import torch.distributed as dist
from datetime import timedelta
from datasets import concatenate_datasets


parser = argparse.ArgumentParser()
parser.add_argument("--per_device_train_batch_size", type=int, default=1)
parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
parser.add_argument("--total_batch_size", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--dataset", type=str, default="prm")
parser.add_argument("--lang", type=str, default="en")
parser.add_argument("--model_name_or_path", type=str, default="")
parser.add_argument("--ds_config", type=str, default="")
parser.add_argument('--local_rank', type=int, default=0)
args = parser.parse_args()


model_name = args.model_name_or_path.split('/')[-1]

if 'mistral-7b-sft' == model_name:
    good_token = '+'
    bad_token = '-'
    step_tag = 'ки'
# elif 'mistral' in model_name.lower():
#     step_tag = ' ######'
elif 'qwen' in model_name.lower() or 'llama' in model_name.lower():
    good_token = ' +'
    bad_token = ' -'
    step_tag = ' \n\n\n\n\n'
print("++++++++++++++",model_name,"+++++++++++++++++++++")
print("++++++++++++++",step_tag,"+++++++++++++++++++++")
model_path = args.model_name_or_path


tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    add_eos_token=False, 
)
if not tokenizer.pad_token:
    tokenizer.pad_token_id = tokenizer.eos_token_id 
    if 'mistral-7b-sft' == model_name:
        tokenizer.pad_token_id = 0 
print("tokenizer.pad_token_id",tokenizer.pad_token_id)
tokenizer.padding_side = "left"  # Allow batched inference


candidate_tokens = tokenizer.encode(f"{good_token}{bad_token}") # [489, 482]
if 'mistral-7b-sft' == model_name:
    candidate_tokens = tokenizer.encode(f"{good_token} {bad_token}") # [648, 387]
if len(candidate_tokens) > 2:
    candidate_tokens = candidate_tokens[1:]
step_tag_id = tokenizer.encode(f"{step_tag}")[-1] # 77425 

print('step_tag_id:',tokenizer.encode(f"{step_tag}"))
print('candidate_tokens:',candidate_tokens)
assert len(candidate_tokens) == 2
assert len(tokenizer.encode(f"{step_tag}")) == 2 or len(tokenizer.encode(f"{step_tag}")) == 1

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    # attn_implementation="flash_attention_2",
)

# lora_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,  # LoRA for causal language modeling task
#     r=8,  # Rank of LoRA
#     lora_alpha=32,  # Alpha scaling factor for LoRA
#     lora_dropout=0.1,  # Dropout rate for LoRA layers
#     target_modules=["q_proj", "v_proj"],  # Apply LoRA to specific layers
# )

# model = get_peft_model(model, lora_config)


# Tokenizing function
def preprocess_function(example):
    input = f"{example['question']} {example['process']}"
    if 'mistral-7b-sft' == model_name:
        process = example['process'].replace(' \n\n\n\n\n ',' ки\n').replace(' \n\n\n\n\n',' ки').strip()
        input = f"{example['question']} {process}"
    tokenized_inputs = tokenizer(
        input, 
        truncation=True, 
        padding="max_length",
        max_length=1024,
    )
    
    
    def find_all_indices(lst, element):
        return [i for i, x in enumerate(lst) if x == element]
    
    length = len(tokenized_inputs['input_ids'])
    # print(length)
    indices = find_all_indices(tokenized_inputs['input_ids'],step_tag_id)
    
    if len(indices) != len(example['label']):
        # print(example)
        example['label'] = example['label'][:len(indices)]
        indices = indices[:len(example['label'])]
    
    assert len(indices) == len(example['label'])
    
    tokenized_inputs['labels'] = [-100] * length
    # tokenized_inputs['labels'] = tokenized_inputs['input_ids']
    tokenized_inputs['attention_mask'] = [1] *length
    
    for i in range(len(indices)):
        if example['label'][i] == '+' or example['label'][i] == 1:
            tokenized_inputs['labels'][indices[i]] = candidate_tokens[0]
        elif example['label'][i] == '-' or example['label'][i] == 0:
            tokenized_inputs['labels'][indices[i]] = candidate_tokens[1]
        else:
            # continue
            raise ValueError('label is wrong')
        tokenized_inputs['attention_mask'][indices[i]] = 0
    # tokenized_inputs['labels'] = [-100] *(length-1) + tokenized_inputs['input_ids'][length-1:]
    
    return tokenized_inputs



DATA_PATH = {
"train": f"/data/math-shepherd_{args.lang}.new.json",
"test": f"/data/phase1_test_{args.lang}.new.json",
}
dataset1 = load_dataset('json',data_files=f"/data/phase1_train_{args.lang}.new.json")
dataset2 = load_dataset('json',data_files=f"/data/phase2_train_{args.lang}.new.json")

dataset = load_dataset('json', data_files=DATA_PATH)
dataset['train'] = concatenate_datasets([dataset['train'], dataset1['train'], dataset2['train']])



print('start processing')
tokenized_datasets = dataset.map(preprocess_function)
tokenized_datasets['train'] = tokenized_datasets['train'].remove_columns(['question','process','label'])
tokenized_datasets['test'] = tokenized_datasets['test'].remove_columns(['question','process','label'])

print('dataset processed')

# Data collator for padding inputs dynamically
data_collator = DataCollatorWithPadding(tokenizer)


fp = f'bs_{args.per_device_train_batch_size}_lr_{args.learning_rate}'
output_path = f'sft/{args.dataset}_{model_name}_{args.lang}_results/{fp}'
repo=f'{args.dataset}_sft_{model_name}_{args.lang}_results_lr_{args.learning_rate}'

print("---------------output_path----------",output_path)

# Training arguments

training_args = TrainingArguments(
    output_dir=output_path,
    evaluation_strategy="steps",  # Evaluate at the end of each epoch
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    do_train=True,
    do_eval=True,
    num_train_epochs=2,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_dir="./logs",
    save_steps=500,
    logging_steps=50,
    save_strategy="steps",
    lr_scheduler_type="linear",
    bf16=True,
    report_to="wandb",  # Set to "wandb" if you are using Weights and Biases for logging
    dataloader_num_workers=16,
    gradient_accumulation_steps=16, 
    deepspeed=args.ds_config,
    ddp_find_unused_parameters=False,
    save_total_limit=3, 
    load_best_model_at_end=True,
    push_to_hub=True,
    hub_strategy="checkpoint",
    hub_model_id=f"bigstupidhats/{repo}",
    hub_private_repo=True,
    resume_from_checkpoint=True, 
)


# Define a custom metric function (e.g., accuracy for binary classification)
def compute_metrics(eval_pred):

    pre, labels = eval_pred
    auc = roc_auc_score(pre[1], pre[0])
    ll = log_loss(pre[1], pre[0])
    acc = accuracy_score(pre[1], pre[0] > 0.5)
    result ={
        'auc': auc, 
        'll': ll, 
        'acc': acc, 
    } 
    print(result)
    return result

def preprocess_logits_for_metrics(logits,labels):
    # print('aa')
    # return logits,labels
    labels_index = torch.argwhere(torch.bitwise_or(labels == candidate_tokens[0], labels == candidate_tokens[1]))
    gold = torch.where(labels[labels_index[:, 0], labels_index[:, 1]] == candidate_tokens[1], 0, 1)
    # labels_index[: , 1] = labels_index[: , 1] - 1
    logits = logits[labels_index[:, 0], labels_index[:, 1]][:, [candidate_tokens[1], candidate_tokens[0]]]
    prob = torch.softmax(logits, dim=-1)
    return prob[:, 1], gold
    

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],  # Replace with a validation set if available
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_state()
trainer.save_model(output_dir=training_args.output_dir)



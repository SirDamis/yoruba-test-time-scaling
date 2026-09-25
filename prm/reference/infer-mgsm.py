import os
import sys

import argparse
import torch
import transformers
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Optional, Dict, Sequence, List
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import json

from tqdm import tqdm
import copy
from torch.cuda.amp import autocast
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    # "prompt_no_input": (
    #     "Below is an instruction that describes a task. "
    #     "Write a response that appropriately completes the request.\n\n"
    #     "### Instruction:\n{query}\n\n### Response:"
    # ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    ),
    "prompt_no_input_v2": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{question}\n\n### Response:"
    ),
}

names = {'en':'English','de':'German','es':'Spanish','fr':'French','ja':'Japanese','ru':'Russian','sw':'Swahili','th':'Thai','zh':'Chinese','bn':'Bengali','te':'Telugu'}

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings, tokenizer: transformers.PreTrainedTokenizer):
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
):
    sources_tokenized = _tokenize_fn(sources, tokenizer)
    input_ids = sources_tokenized["input_ids"]
    return dict(input_ids=input_ids, labels=copy.deepcopy(input_ids))


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer,model_name,lang):
        super(SupervisedDataset, self).__init__()
        with open(data_path, 'r') as f:
            dataset_for_eval = json.load(f)
        if 'octopus' in model_name:
            name = names[lang]
            prompt_no_input = (
                  "Below is an instruction that describes a task. "
                    f"Write a response that appropriately completes the request in {name}. Please answer in {name}.\n\n"
                    "### Instruction:\n{query}\n\n### Response:"
                )
            sources = [prompt_no_input.format(query=item['instruction']) for item in dataset_for_eval]
        else:
            sources = [PROMPT_DICT["prompt_no_input"].format_map(item)for item in dataset_for_eval]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"] 
        self.labels = data_dict["labels"] 

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i], id=i)

def padding(inputs, padding_token, cutoff = None):
    num_elems = len(inputs)
    if cutoff is None:
        cutoff = max([len(item) for item in inputs])
    else:
        cutoff = min(max([len(item) for item in inputs]), cutoff)
    
    tokens = torch.ones(num_elems, cutoff).long().to(inputs[0].device) * padding_token
    for i in range(num_elems):
        toks = inputs[i]
        length = min(cutoff, len(toks))
        tokens[i, -length:] = toks[-length:]
    return tokens

def sequence_gather(s, world_size, pad_tok_id):
    local_size = torch.tensor(s.size(), device=s.device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_length = max(size[1] for size in all_sizes)
    length_diff = max_length.item() - local_size[1].item()
    if length_diff:
        pad_size = (*s.shape[:-1], length_diff)
        padding = torch.ones(pad_size, device=s.device, dtype=s.dtype) * pad_tok_id
        s = torch.concat((s, padding), dim = -1)
    gathered_s = [torch.ones_like(s)*pad_tok_id for _ in range(world_size)]
    dist.all_gather(gathered_s, s)

    return gathered_s

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, ids = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", 'id'))
        input_ids = padding(input_ids, self.tokenizer.pad_token_id, cutoff = 256)
        labels = padding(labels, IGNORE_INDEX, cutoff = 256)

        return dict(
            input_ids=input_ids,
            labels=labels,
            id=torch.tensor(ids).to(input_ids.device),
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_path,model,lang) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_path,model_name=model,lang=lang)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return eval_dataset, data_collator


def main(args):


    torch.manual_seed(args.seed)
    world_size = torch.cuda.device_count()
    base_model = args.base_model
    data_path = f"/bask/homes/f/fksv3157/xngs6460-languages/weixuan/data/MGSM/mgsm_{args.lang}.json"
    batch_size = args.batch_size

    model = transformers.AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
        )
    # model.half()
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model)

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if "llama" in base_model:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )
    tokenizer.truncation_side = 'left'

    model.to(torch.cuda.current_device())
    model.eval()



    eval_dataset, data_collator = make_supervised_data_module(tokenizer, data_path,base_model,args.lang)
    return_seq_num = 1
    for tempera in [0.7]:
        # sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, num_replicas=world_size, shuffle=False)
        dataloader = DataLoader(
            eval_dataset, 
            shuffle=False, 
            collate_fn=data_collator, 
            # batch_size=batch_size,
            # sampler=sampler,
            # drop_last=True,
        )
        generation_config = GenerationConfig(
            # temperature=0.8 if args.diverse_beam > 1 else 1.0,
            temperature=tempera,
            # num_beam_groups=args.diverse_beam,
            # diversity_penalty=1.0,
            do_sample=True,
            num_beams=return_seq_num,
            max_new_tokens=256,
            num_return_sequences=return_seq_num,
        )

        all_outputs = []
        # for step,batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        for step,inp in tqdm(enumerate(dataloader), total=len(dataloader)):

            input_ids = inp['input_ids'].to(model.device)
            attention_mask = inp['attention_mask'].to(model.device)
            with torch.no_grad():
                generation_output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    pad_token_id=tokenizer.pad_token_id,
                    # return_dict_in_generate=True,
                    # synced_gpus=True,
                )
            # print(generation_output)
            s = tokenizer.batch_decode(generation_output)[0]
            ori = tokenizer.batch_decode(input_ids)[0]
            res = s.split(ori)[-1]
            temp = {'instruction':ori,'generation':res}
            # print(temp)
            all_outputs.append(temp)
   
        out_f = open(args.out_path + f'/raw_generation_{tempera}_{args.seed}.json', 'w')
        json.dump(all_outputs, out_f,indent=4, ensure_ascii=False)
        out_f.close()

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--base_model", default="", type=str, help="model path")
    parser.add_argument("--lang", default="", type=str, help="language")
    parser.add_argument("--data_path", default="", type=str, help="config path")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--port", type=int, default=0, help="batch size")
    parser.add_argument("--diverse_beam", type=int, default=1, help="batch size")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    parser.add_argument("--out_path", default="", type=str, help="config path")
    args = parser.parse_args()

    # local_rank = int(os.environ["LOCAL_RANK"])
    # main(local_rank, args)
    main(args)

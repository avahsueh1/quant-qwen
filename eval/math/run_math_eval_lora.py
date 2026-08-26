#!/usr/bin/env python3
import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_INSTRUCTION = """You are being evaluated on mathematical reasoning.
Solve the problem using only the information in the prompt and standard mathematics.
Show enough reasoning to make your conclusion auditable.
If the information is insufficient to determine a unique answer, say so explicitly and justify why.
Do not assume unstated facts merely to force a numerical answer."""

def set_determinism(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

def load_jsonl(path):
    rows=[]
    with Path(path).open('r',encoding='utf-8') as f:
        for n,line in enumerate(f,1):
            line=line.strip()
            if not line: continue
            try: rows.append(json.loads(line))
            except json.JSONDecodeError as e: raise ValueError(f'Invalid JSON on line {n}: {e}')
    return rows

def completed_ids(path):
    p=Path(path); done=set()
    if not p.exists(): return done
    with p.open('r',encoding='utf-8') as f:
        for line in f:
            try:
                row=json.loads(line)
                if 'id' in row: done.add(row['id'])
            except Exception: pass
    return done

def make_prompt(item):
    return (SYSTEM_INSTRUCTION + '\n\nProblem ID: '+item['id'] + '\nProblem: '+item['title'] + '\n\n'+item['prompt'].strip()+'\n\nAnswer:')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',default='Qwen/Qwen3-4B-Base')
    ap.add_argument('--adapter',required=True)
    ap.add_argument('--benchmark',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--max-new-tokens',type=int,default=512)
    ap.add_argument('--limit',type=int)
    ap.add_argument('--start-id')
    ap.add_argument('--dtype',choices=['bf16','fp16','fp32','auto'],default='bf16')
    ap.add_argument('--trust-remote-code',action='store_true')
    args=ap.parse_args()

    set_determinism(args.seed)
    dtype_map={'bf16':torch.bfloat16,'fp16':torch.float16,'fp32':torch.float32,'auto':'auto'}

    print('Model:', args.model)
    print('Adapter:', args.adapter)
    print('Loading tokenizer:',args.model)
    tok=AutoTokenizer.from_pretrained(args.model,trust_remote_code=args.trust_remote_code)
    print('Loading model:',args.model)
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=dtype_map[args.dtype],device_map='auto',trust_remote_code=args.trust_remote_code)
    print('Loading LoRA adapter:', args.adapter)
    model=PeftModel.from_pretrained(model,args.adapter,is_trainable=False)
    model.eval()
    if tok.pad_token_id is None: tok.pad_token_id=tok.eos_token_id

    items=load_jsonl(args.benchmark)
    if args.start_id:
        ids=[x['id'] for x in items]
        if args.start_id not in ids: raise ValueError(f'{args.start_id} not found')
        items=items[ids.index(args.start_id):]
    done=completed_ids(args.output)
    items=[x for x in items if x['id'] not in done]
    if args.limit is not None: items=items[:args.limit]

    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    print('Questions to run:',len(items))
    run_started=datetime.now(timezone.utc).isoformat()

    with out.open('a',encoding='utf-8') as f:
        for idx,item in enumerate(items,1):
            prompt=make_prompt(item)
            enc=tok(prompt,return_tensors='pt')
            input_ids=enc['input_ids'].to(model.device)
            attention_mask=enc.get('attention_mask')
            if attention_mask is not None: attention_mask=attention_mask.to(model.device)
            t0=time.time()
            with torch.inference_mode():
                generated=model.generate(input_ids=input_ids,attention_mask=attention_mask,max_new_tokens=args.max_new_tokens,do_sample=False,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
            completion_ids=generated[0,input_ids.shape[1]:]
            response=tok.decode(completion_ids,skip_special_tokens=True).strip()
            # Base models sometimes continue by inventing the next benchmark problem.
            # Keep only the response to the current problem.
            if "\nProblem ID:" in response:
                response = response.split("\nProblem ID:", 1)[0].strip()
            row={'id':item['id'],'title':item['title'],'skills':item.get('skills',[]),'benchmark_version':item.get('benchmark_version'),'model':args.model,'adapter':args.adapter,'seed':args.seed,'do_sample':False,'max_new_tokens':args.max_new_tokens,'dtype':args.dtype,'prompt':item['prompt'],'response':response,'input_tokens':int(input_ids.shape[1]),'output_tokens':int(completion_ids.shape[0]),'elapsed_seconds':round(time.time()-t0,4),'run_started_utc':run_started,'completed_utc':datetime.now(timezone.utc).isoformat()}
            f.write(json.dumps(row,ensure_ascii=False)+'\n'); f.flush()
            print(f"[{idx}/{len(items)}] {item['id']} {row['output_tokens']} tok {row['elapsed_seconds']}s")

if __name__=='__main__': main()

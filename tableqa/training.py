"""Single-GPU BF16 LoRA with per-example target normalization and typed losses."""
import json
import random
from collections import Counter
from pathlib import Path
from .data import digest


def balance_and_budget(encoded, seed=42, target_token_budget=None, sampling_policy='replacement'):
    if sampling_policy not in {'replacement','coverage'}:raise ValueError('Unknown sampling policy')
    by_kind = {k:[x for x in encoded if x["kind_id"] == k] for k in (0,1)}
    rng=random.Random(seed)
    if all(by_kind.values()):
        n=max(map(len,by_kind.values()))
        if sampling_policy=='replacement':
            # Historical developmental runs: retained explicitly for audit/replay.
            encoded=[rng.choice(by_kind[k]) for _ in range(n) for k in (0,1)]
        else:
            # Each original record appears once before upsampling a smaller kind.
            pools={}
            for k in (0,1):
                pool=[]
                while len(pool)<n:
                    cycle=list(by_kind[k]);rng.shuffle(cycle)
                    pool+=cycle[:n-len(pool)]
                pools[k]=pool
            encoded=[pools[k][i] for i in range(n) for k in (0,1)]
    else:
        encoded=list(encoded)
    rng.shuffle(encoded)
    selected=[]; count=0
    for row in encoded:
        tokens=sum(v != -100 for v in row["labels"])
        if target_token_budget is not None and count+tokens>target_token_budget: continue
        selected.append(row); count+=tokens
    if not selected: raise ValueError("Empty training set after balancing/token budget")
    return selected,count


def align_accumulation_batches(encoded, multiple, token_budget=None):
    """Avoid older Trainer versions dropping the last partial accumulation batch."""
    if multiple<1: raise ValueError('Effective batch size must be positive')
    rows=list(encoded); padding=(-len(rows))%multiple
    rows += [encoded[i%len(encoded)] for i in range(padding)]
    tokens=lambda values:sum(sum(t!=-100 for t in r['labels']) for r in values)
    dropped=0
    while token_budget is not None and tokens(rows)>token_budget and rows:
        rows=rows[:-multiple]; dropped+=multiple
    if not rows: raise ValueError('Token budget cannot fit one complete accumulation batch')
    return rows,{'padding_records':padding,'dropped_budget_records':dropped,'supervised_tokens_per_epoch':tokens(rows)}


def encode_record(record, tokenizer, max_length):
    prefix = tokenizer.apply_chat_template(record["messages"], tokenize=False, add_generation_prompt=True)
    # Encode the target separately so the prompt/target boundary is exact.
    x = tokenizer.encode(prefix, add_special_tokens=False)
    y = tokenizer.encode(record["target"], add_special_tokens=False) + [tokenizer.eos_token_id]
    if len(x) + len(y) > max_length:
        raise ValueError(f"Overlength training example {record['id']}; no silent truncation")
    if not x or not y: raise ValueError("Empty prompt/target")
    return {"input_ids": x + y, "attention_mask": [1]*(len(x)+len(y)),
            "labels": [-100]*len(x)+y, "kind_id": 0 if record["kind"] == "block" else 1,
            "source_key": (record.get('dataset','unknown'),record['id']),"source_label":record.get('sampling_label')}


def attach_loss_weights(encoded,policy='none'):
    """Mean-one weights for task/kind/verification-label strata; no new sampling."""
    if policy not in {'none','task_label'}:raise ValueError('Unknown loss balancing policy')
    if policy=='none':return encoded,[]
    groups=Counter()
    def key(row):
        dataset=row['source_key'][0];label=row.get('source_label') if dataset=='tabfact' else None
        if dataset=='tabfact' and (type(label) is not int or label not in (0,1)):
            raise ValueError('Task/label balancing requires explicit training-only TabFact labels')
        return dataset,row['kind_id'],label
    for row in encoded:groups[key(row)]+=1
    datasets={k[0] for k in groups};kinds={d:{k[1] for k in groups if k[0]==d} for d in datasets}
    labels={(d,k):{r[2] for r in groups if r[:2]==(d,k)} for d in datasets for k in kinds[d]}
    weights={s:len(encoded)/(len(datasets)*len(kinds[s[0]])*len(labels[s[:2]])*n) for s,n in groups.items()}
    result=[{**row,'sample_weight':weights[key(row)]} for row in encoded]
    report=[{'dataset':s[0],'kind':'block' if s[1]==0 else 'answer','label':s[2],
             'records':groups[s],'weight':w,'weighted_mass':w*groups[s]} for s,w in sorted(weights.items())]
    return result,report


def prepare_training_data(config,records,tokenizer):
    """Exact pre-GPU sampling audit. Retains the existing sampler/ordering."""
    encoded=[encode_record(r,tokenizer,config.get('max_seq_length',4096)) for r in records]
    initial_keys={x['source_key'] for x in encoded}
    policy=config.get('sampling_policy','replacement')
    if policy == 'frozen':
        if config.get('target_token_budget') is not None:
            raise ValueError('Frozen schedules cannot be cropped by a second token budget')
        if config.get('frozen_records_sha256') != digest(records):
            raise ValueError('Frozen record schedule identity mismatch')
        counts=Counter(r['kind_id'] for r in encoded)
        if len(counts)==2 and counts[0]!=counts[1]:
            raise ValueError('Frozen two-kind schedule must balance block and answer records')
        multiple=config.get('micro_batch',1)*config.get('gradient_accumulation_steps',16)
        if len(encoded)%multiple:
            raise ValueError('Frozen schedule must already align complete optimizer batches')
        tokens=sum(sum(t!=-100 for t in row['labels']) for row in encoded)
        if tokens != config.get('frozen_supervised_tokens_per_epoch'):
            raise ValueError('Frozen schedule target tokens differ from preregistered count')
    else:
        encoded,_=balance_and_budget(encoded,config.get('seed',42),config.get('target_token_budget'),policy)
    balanced_count=len(encoded)
    encoded,alignment=align_accumulation_batches(encoded,config.get('micro_batch',1)*config.get('gradient_accumulation_steps',16),config.get('target_token_budget'))
    encoded,loss_strata=attach_loss_weights(encoded,config.get('loss_balance','none'))
    exposures={}; tokens_by_kind=Counter()
    for r in encoded:
        key=(*r['source_key'],r['kind_id']); tokens=sum(t!=-100 for t in r['labels'])
        item=exposures.setdefault(key,{'dataset':key[0],'id':key[1],
                                      'kind':'block' if key[2]==0 else 'answer','record_exposures':0,'target_tokens':0})
        item['record_exposures']+=1; item['target_tokens']+=tokens
        tokens_by_kind[item['kind']]+=tokens
    retained={x['source_key'] for x in encoded}
    stats={'sampling_policy':policy,'records_before_balancing':len(records),'records_after_balancing':balanced_count,
           'loss_balance':config.get('loss_balance','none'),'loss_weight_strata':loss_strata,
           'records_after_alignment':len(encoded),**alignment,
           'requested_target_token_budget':config.get('target_token_budget'),
           'source_samples_before_balancing':len(initial_keys),'source_samples_after_alignment':len(retained),
           'source_samples_by_dataset':dict(Counter(k[0] for k in retained)),
           'target_tokens_by_kind':dict(tokens_by_kind),
           'retained_source_ids_sha256':digest(sorted(retained)),
           'sampling_sequence_sha256':digest([(r['source_key'],r['kind_id'],r['labels']) for r in encoded]),
           'source_exposure':[exposures[k] for k in sorted(exposures)]}
    stats['input_tokens_per_epoch']=sum(sum(t==-100 for t in row['labels']) for row in encoded)
    stats['nonpadding_tokens_per_epoch']=sum(len(row['input_ids']) for row in encoded)
    return encoded,stats


def make_typed_trainer_class(config,has_both_kinds):
    """Lazy optional dependencies; explicit mean over a complete accumulation.

    Transformers 4.46.3 delegates accumulation to its loop while Accelerator's
    backward divisor is 1. A custom per-example mean must therefore divide by
    the configured number of microbatches itself. Ignore token-count kwargs:
    this objective averages examples, not all tokens in a combined batch.
    """
    import torch
    import torch.nn.functional as F
    from transformers import Trainer
    class TypedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            kinds=inputs.pop("kind_id"); labels=inputs.pop("labels");sample_weights=inputs.pop('sample_weight')
            outputs=model(**inputs)
            logits=outputs.logits[:,:-1].contiguous().float(); targets=labels[:,1:].contiguous()
            per_token=F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1),
                                     reduction="none",ignore_index=-100).view_as(targets)
            mask=targets.ne(-100); per_example=per_token.sum(-1)/mask.sum(-1).clamp(min=1)
            weights=torch.where(kinds.eq(1),float(config.get("answer_weight",1)),1.)
            factor=2. if has_both_kinds else 1.
            loss=(per_example*weights*sample_weights).mean()*factor
            return (loss,outputs) if return_outputs else loss

        def training_step(self,model,inputs,num_items_in_batch=None):
            if self.use_apex or self.is_deepspeed_enabled or self.args.n_gpu>1:
                raise RuntimeError('Explicit mean trainer supports ordinary single-device Accelerate only')
            model.train()
            if hasattr(self.optimizer,'train') and callable(self.optimizer.train):self.optimizer.train()
            inputs=self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():loss=self.compute_loss(model,inputs)
            accumulation=self.args.gradient_accumulation_steps
            # Accelerator.backward divides by its own setting. Cancel that
            # divisor explicitly, then normalize by our complete microbatch count.
            accelerator_divisor=self.accelerator.gradient_accumulation_steps
            self.accelerator.backward(loss*(accelerator_divisor/accumulation))
            # Trainer sums these detached values across microbatches for logging.
            return loss.detach()/accumulation
    return TypedTrainer


def train(config, records, output, validation=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model
    if any(r.get("split") != "train" for r in records): raise ValueError("Non-training record supplied")
    if not records: raise ValueError("Empty training set")
    if validation and any(r.get("split") != "validation" for r in validation):
        raise ValueError("Validation records need explicit validation split")
    set_seed(config.get("seed",42))
    if not torch.cuda.is_available(): raise RuntimeError("GPU training requires CUDA; use demo for CPU verification")
    if torch.cuda.device_count()!=1: raise RuntimeError('Single-GPU trainer: select one GPU with CUDA_VISIBLE_DEVICES')
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config.get("revision"))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    maximum = config.get("max_seq_length",4096)
    encoded,preparation = prepare_training_data(config,records,tokenizer)
    val = [encode_record(r,tokenizer,maximum) for r in validation] if validation else None
    has_both_kinds=len({x["kind_id"] for x in encoded})==2
    model = AutoModelForCausalLM.from_pretrained(config["model"], revision=config.get("revision"),
                torch_dtype=torch.bfloat16)
    model.config.use_cache=False
    if config.get("initial_adapter"):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model,config["initial_adapter"],is_trainable=True)
    else:
        model = get_peft_model(model,LoraConfig(task_type="CAUSAL_LM",r=config.get("lora_rank",32),
                    lora_alpha=config.get("lora_alpha",64),lora_dropout=.05,target_modules="all-linear"))
    if hasattr(model,"enable_input_require_grads"): model.enable_input_require_grads()
    def collate(batch):
        length=max(len(x["input_ids"]) for x in batch); result={}
        for key,pad in (("input_ids",tokenizer.pad_token_id),("attention_mask",0),("labels",-100)):
            result[key]=torch.tensor([x[key]+[pad]*(length-len(x[key])) for x in batch])
        result["kind_id"]=torch.tensor([x["kind_id"] for x in batch])
        result['sample_weight']=torch.tensor([x.get('sample_weight',1.) for x in batch],dtype=torch.float32)
        return result
    TypedTrainer=make_typed_trainer_class(config,has_both_kinds)
    args=TrainingArguments(output_dir=str(output),num_train_epochs=config.get("max_epochs",3),
        per_device_train_batch_size=config.get("micro_batch",1),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps",16),
        learning_rate=config.get("learning_rate",2e-4),bf16=True,
        gradient_checkpointing=True,logging_steps=10,save_strategy="epoch",
        eval_strategy="epoch" if val else "no",load_best_model_at_end=bool(val),
        metric_for_best_model="eval_loss",greater_is_better=False,
        save_total_limit=2,report_to=[],seed=config.get("seed",42),
        remove_unused_columns=False)
    trainer=TypedTrainer(model=model,args=args,train_dataset=encoded,eval_dataset=val,data_collator=collate)
    trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint"))
    preparation.update(gradient_normalization='explicit-microbatch-mean-v1',
        configured_gradient_accumulation_steps=args.gradient_accumulation_steps,
        accelerator_backward_divisor=trainer.accelerator.gradient_accumulation_steps)
    trainer.save_model(str(output)); tokenizer.save_pretrained(str(output))
    Path(output,"training_config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    Path(output,"training_history.json").write_text(json.dumps(trainer.state.log_history,indent=2),encoding="utf-8")
    exposure=preparation.pop('source_exposure')
    Path(output,'training_exposure.json').write_text(json.dumps({'summary':preparation,'per_source':exposure},indent=2),encoding='utf-8')
    return {**preparation,"peak_allocated_bytes":torch.cuda.max_memory_allocated()}

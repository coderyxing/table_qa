"""Measure legacy and corrected gradients against the same full-batch loss.

Uses a tiny CPU model and the actual installed Trainer/Accelerator. Loads only
the project's own legacy nested trainer class via AST, not arbitrary plans.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import types


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('legacy_training_py');parser.add_argument('output')
    args=parser.parse_args();output=Path(args.output)
    if output.exists():raise ValueError('Preserve existing gradient audit')
    import torch
    import torch.nn.functional as F
    from transformers import Trainer,TrainingArguments
    from tableqa.training import make_typed_trainer_class
    from tableqa.cli import metadata
    source=Path(args.legacy_training_py);module=ast.parse(source.read_text(encoding='utf-8'))
    function=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='train')
    nested=next(n for n in function.body if isinstance(n,ast.ClassDef) and n.name=='TypedTrainer')
    scope={'Trainer':Trainer,'torch':torch,'F':F,'config':{'answer_weight':1},'has_both_kinds':False}
    exec(compile(ast.Module(body=[nested],type_ignores=[]),str(source), 'exec'),scope)
    classes={'legacy':scope['TypedTrainer'],'corrected':make_typed_trainer_class({'answer_weight':1},False)}
    class Toy(torch.nn.Module):
        def __init__(self):super().__init__();self.embedding=torch.nn.Embedding(7,5);self.linear=torch.nn.Linear(5,7)
        def forward(self,input_ids,attention_mask=None,**loss_kwargs):return types.SimpleNamespace(logits=self.linear(self.embedding(input_ids)))
    rows=[]
    for accumulation in [2,16]:
        for name,cls in classes.items():
            torch.manual_seed(42);model=Toy();reference=copy.deepcopy(model)
            def batch(ids):
                x=torch.tensor([[0,1,2],[0,3,4]])[[i%2 for i in ids]]
                return {'input_ids':x,'attention_mask':torch.ones_like(x),
                        'labels':x.clone().masked_fill(torch.arange(3).unsqueeze(0)==0,-100),
                        'kind_id':torch.ones(len(ids),dtype=torch.long),'sample_weight':torch.ones(len(ids))}
            with tempfile.TemporaryDirectory() as tmp:
                trainer=cls(model=model,args=TrainingArguments(output_dir=tmp,use_cpu=True,
                    gradient_accumulation_steps=accumulation,per_device_train_batch_size=1,report_to=[]))
                full_loss=trainer.compute_loss(reference,batch(list(range(accumulation))));full_loss.backward()
                reference_gradient=torch.cat([p.grad.flatten() for p in reference.parameters()]);logged=0
                for i in range(accumulation):
                    logged+=float(trainer.training_step(model,batch([i]),num_items_in_batch=torch.tensor(2*accumulation)))
                actual=torch.cat([p.grad.flatten() for p in model.parameters()])
                ratio=float((actual@reference_gradient)/(reference_gradient@reference_gradient))
                rows.append({'implementation':name,'configured_accumulation':accumulation,
                    'accelerator_backward_divisor':trainer.accelerator.gradient_accumulation_steps,
                    'model_accepts_loss_kwargs':trainer.model_accepts_loss_kwargs,
                    'measured_gradient_scale_vs_full_batch':ratio,
                    'max_absolute_gradient_error':float((actual-reference_gradient).abs().max()),
                    'logged_loss_sum':logged,'full_batch_loss':float(full_loss.detach())})
                if name=='corrected' and not torch.allclose(actual,reference_gradient,atol=1e-7,rtol=1e-6):
                    raise ValueError('Corrected gradient is not the full-batch mean')
    report={'probe_only_not_model_accuracy':True,'legacy_source':str(source),
            'legacy_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'current_runtime':metadata({}),'measurements':rows}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()

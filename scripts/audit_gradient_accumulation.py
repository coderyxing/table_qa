# 脚本用途：用玩具模型比较历史实现、修复实现与完整批次的梯度尺度。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
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


# 解析当前脚本参数，按模块说明执行本次冻结阶段并保存产物。
# 相关调用：output.write_text、source.read_text、make_typed_trainer_class、hashlib.sha256(source.read_bytes()).hexdigest。
# 返回形式含：字段 input_ids、attention_mask、labels、kind_id、sample_weight。
# 异常约束：Preserve existing gradient audit
def main():
    # 定义本入口的参数解析器；位置参数、默认值和允许取值由相邻 add_argument 明确声明。
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('legacy_training_py');parser.add_argument('output')
    # 解析本次命令的实际参数，后续样本路径和实验开关均由此确定。
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
    # Toy：提供可精确比较梯度与参数更新的小型模型。
    class Toy(torch.nn.Module):
        # 初始化 Toy 的依赖及持久状态；具体预算和身份字段见赋值。
        def __init__(self):super().__init__();self.embedding=torch.nn.Embedding(7,5);self.linear=torch.nn.Linear(5,7)
        # 计算测试模型的前向结果，供梯度与更新等价检查。
        # 输入：input_ids；attention_mask；**loss_kwargs。
        def forward(self,input_ids,attention_mask=None,**loss_kwargs):return types.SimpleNamespace(logits=self.linear(self.embedding(input_ids)))
    rows=[]
    for accumulation in [2,16]:
        for name,cls in classes.items():
            torch.manual_seed(42);model=Toy();reference=copy.deepcopy(model)
            # 构造玩具训练模型使用的完整或微批次输入。
            # 输入：ids。
            # 返回形式含：字段 input_ids、attention_mask、labels、kind_id、sample_weight。
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

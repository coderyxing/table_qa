# 测试范围：验证对应模块的数据隔离、输入约束与历史回归行为。
# 运行方式：python -m unittest discover -s tests -p test_trainer_gradient.py -v
# 阅读顺序：先看样本或替身模型的构造，再看每个用例前的预期约束。
"""Actual optional Trainer integration: microbatches must equal a full batch."""
import copy
import importlib.util
import tempfile
import types
import unittest

AVAILABLE=all(importlib.util.find_spec(p) is not None for p in ['torch','transformers','accelerate'])


# TrainerGradientTests：集中验证本模块的协议与回归场景；各用例独立构造输入并核对预期结果。
@unittest.skipUnless(AVAILABLE,'Optional actual training dependencies are absent')
class TrainerGradientTests(unittest.TestCase):
    # 准备本组测试的独立样本与执行状态，避免用例之间共享可变结果。
    def setUp(self):
        import torch
        self.torch=torch;torch.manual_seed(42)
        # Toy：集中验证本模块的协议与回归场景；各用例独立构造输入并核对预期结果。
        class Toy(torch.nn.Module):
            # 初始化 Toy 的依赖及持久状态；具体预算和身份字段见赋值。
            def __init__(self):
                super().__init__();self.embedding=torch.nn.Embedding(7,5);self.linear=torch.nn.Linear(5,7)
            # 计算测试模型的前向结果，供梯度与更新等价检查。
            # 输入：input_ids；attention_mask；**loss_kwargs。
            def forward(self,input_ids,attention_mask=None,**loss_kwargs):
                return types.SimpleNamespace(logits=self.linear(self.embedding(input_ids)))
        self.model=Toy()

    # 构造带目标、类型与权重的训练行，供梯度等价测试使用。
    # 返回形式含：列表记录。
    def rows(self):
        return [{'input_ids':[0,1,2],'attention_mask':[1,1,1],'labels':[-100,1,2],
                 'kind_id':0,'sample_weight':.5},
                {'input_ids':[0,3,4],'attention_mask':[1,1,1],'labels':[-100,-100,4],
                 'kind_id':1,'sample_weight':1.5}]

    # 按当前批次最长序列右侧补齐输入、注意力和标签，补齐标签使用 -100。
    # 输入：rows。
    def collate(self,rows):return {k:self.torch.tensor([r[k] for r in rows]) for k in rows[0]}

    # 建立受控的玩具 Trainer，比较完整批次与梯度累积的参数更新。
    # 输入：model（生成后端）；root（工作根目录）；micro；accumulation。
    # 相关调用：make_typed_trainer_class({'answer_weight': 2}, True)、make_typed_trainer_class。
    def trainer(self,model,root,micro=1,accumulation=2):
        from transformers import TrainingArguments
        from tableqa.training import make_typed_trainer_class
        args=TrainingArguments(output_dir=root,use_cpu=True,per_device_train_batch_size=micro,
             gradient_accumulation_steps=accumulation,max_steps=1,optim='sgd',learning_rate=.1,
             lr_scheduler_type='constant',max_grad_norm=0,weight_decay=0,
             report_to=[],save_strategy='no',logging_strategy='no',disable_tqdm=True,
             remove_unused_columns=False)
        return make_typed_trainer_class({'answer_weight':2},True)(
            model=model,args=args,train_dataset=self.rows(),data_collator=self.collate)

    # 回归用例：test_actual_accumulated_gradient_and_logged_loss_equal_full_batch
    # 应成立：torch.allclose(actual, reference_gradient, atol=1e-07, rtol=1e-06)
    # 应在浮点容差内相等：logged，float(loss.detach())
    def test_actual_accumulated_gradient_and_logged_loss_equal_full_batch(self):
        torch=self.torch;reference=copy.deepcopy(self.model)
        with tempfile.TemporaryDirectory() as root:
            trainer=self.trainer(self.model,root)
            loss=trainer.compute_loss(reference,self.collate(self.rows()));loss.backward()
            reference_gradient=torch.cat([p.grad.flatten() for p in reference.parameters()])
            logged=0
            for row in self.rows():
                # kwargs carries a TOKEN count, not our per-example denominator.
                logged+=float(trainer.training_step(self.model,self.collate([row]),num_items_in_batch=torch.tensor(3)))
            actual=torch.cat([p.grad.flatten() for p in self.model.parameters()])
            self.assertTrue(torch.allclose(actual,reference_gradient,atol=1e-7,rtol=1e-6))
            self.assertAlmostEqual(logged,float(loss.detach()),delta=1e-6)

    # 回归用例：test_actual_training_loop_optimizer_update_matches_full_batch
    # 应成立：torch.allclose(left, right, atol=1e-07, rtol=1e-06)
    def test_actual_training_loop_optimizer_update_matches_full_batch(self):
        torch=self.torch;full=copy.deepcopy(self.model)
        with tempfile.TemporaryDirectory() as first,tempfile.TemporaryDirectory() as second:
            a=self.trainer(self.model,first,micro=1,accumulation=2)
            b=self.trainer(full,second,micro=2,accumulation=1)
            a.train();b.train()
            for left,right in zip(self.model.parameters(),full.parameters()):
                self.assertTrue(torch.allclose(left,right,atol=1e-7,rtol=1e-6))


if __name__=='__main__':unittest.main()

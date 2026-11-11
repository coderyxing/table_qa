"""Actual optional Trainer integration: microbatches must equal a full batch."""
import copy
import importlib.util
import tempfile
import types
import unittest

AVAILABLE=all(importlib.util.find_spec(p) is not None for p in ['torch','transformers','accelerate'])


@unittest.skipUnless(AVAILABLE,'Optional actual training dependencies are absent')
class TrainerGradientTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch=torch;torch.manual_seed(42)
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__();self.embedding=torch.nn.Embedding(7,5);self.linear=torch.nn.Linear(5,7)
            def forward(self,input_ids,attention_mask=None,**loss_kwargs):
                return types.SimpleNamespace(logits=self.linear(self.embedding(input_ids)))
        self.model=Toy()

    def rows(self):
        return [{'input_ids':[0,1,2],'attention_mask':[1,1,1],'labels':[-100,1,2],
                 'kind_id':0,'sample_weight':.5},
                {'input_ids':[0,3,4],'attention_mask':[1,1,1],'labels':[-100,-100,4],
                 'kind_id':1,'sample_weight':1.5}]

    def collate(self,rows):return {k:self.torch.tensor([r[k] for r in rows]) for k in rows[0]}

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

    def test_actual_training_loop_optimizer_update_matches_full_batch(self):
        torch=self.torch;full=copy.deepcopy(self.model)
        with tempfile.TemporaryDirectory() as first,tempfile.TemporaryDirectory() as second:
            a=self.trainer(self.model,first,micro=1,accumulation=2)
            b=self.trainer(full,second,micro=2,accumulation=1)
            a.train();b.train()
            for left,right in zip(self.model.parameters(),full.parameters()):
                self.assertTrue(torch.allclose(left,right,atol=1e-7,rtol=1e-6))


if __name__=='__main__':unittest.main()

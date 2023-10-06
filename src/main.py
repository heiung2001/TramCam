import numpy as np
import os
import argparse

from transformers import TrainingArguments
from models.xlm_roberta import MRCwithRoberta
from transformers import Trainer
from utils import data_loader
from datasets import load_metric

from peft.mapping import get_peft_model, get_peft_config
from peft.tuners.lora import LoraConfig
from peft.utils.peft_types import PeftType, TaskType


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.required_grad:
            trainable_params += param.numel()
            print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}") 


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # general
    parser.add_argument('-base_model', default='xlm-roberta-large', help='Model to fine-tune')
    parser.add_argument('-use_fp16', default=True, choices=[True, False], help="Whether to use fp16 or not")
    parser.add_argument('-save_strategy', default='epoch', choices=['epoch', 'steps'])
    parser.add_argument('-eval_strategy', default='epoch', choices=['epoch, steps'])
    parser.add_argument('-logging_steps', default=100, type=int, help='Logging after # steps')

    # hyperparameters
    parser.add_argument('-num_epochs', default=5, type=int, help="Number of epochs")
    parser.add_argument('-train_batch_size', default=8, type=int)
    parser.add_argument('-eval_batch_size', default=8, type=int)
    parser.add_argument('-gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('-num_workers', default=2, type=int)
    parser.add_argument('-metric_for_best_model', default='f1', choices=['f1', 'EM'])
    parser.add_argument('-eval_steps', default=5, type=int)

    # optimizer
    parser.add_argument('-learning_rate', default=1e-4, type=float)
    parser.add_argument('-weight_decay', default=1e-2, type=float)
    parser.add_argument('-warmup_ratio', default=5e-2, type=float)

    # file, dir, path
    parser.add_argument('-pretrained_storage_path', default='./models/cache')
    parser.add_argument('-train_path', default='data/processed/train.dataset', help="Using folder created by huggingface's dataset")
    parser.add_argument('-valid_path', default='data/processed/valid.dataset', help="Using folder created by huggingface's dataset")
    parser.add_argument('-logging_path', default='./log')

    # lora
    parser.add_argument('-lora_rank', default=1, type=int)
    parser.add_argument('-lora_dropout', default=0.1, type=float)
    parser.add_argument('-lora_alpha', default=16)
    parser.add_argument('-lora_bias', default='none')

    args = parser.parse_args()

    model = MRCwithRoberta.from_pretrained(pretrained_model_name_or_path=args.base_model,
                                           cache_dir=args.pretrained_storage_path,
                                           #local_files_only=True
                                           )
    print(model)
    print(model.config)

    peft_config = LoraConfig(
        task_type='CAUSAL_LM',
        r=args.lora_rank,
        target_modules=["query_key_value"],
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias
    )
    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)


    train_dataset, valid_dataset = data_loader.get_dataloader(
        train_path=args.train_path,
        valid_path=args.valid_path
    )

    training_args = TrainingArguments("model-bin/test",
                                      do_train=True,
                                      do_eval=True,
                                      num_train_epochs=args.num_epochs,
                                      learning_rate=args.learning_rate,
                                      warmup_ratio=args.warmup_ratio,
                                      weight_decay=args.weight_decay,
                                      per_device_train_batch_size=args.train_batch_size,
                                      per_device_eval_batch_size=args.eval_batch_size,
                                      gradient_accumulation_steps=args.gradient_accumulation_steps,
                                      logging_dir=args.logging_path,
                                      logging_steps=args.logging_steps,
                                      label_names=['start_positions',
                                                   'end_positions',
                                                   'span_answer_ids',
                                                   'input_ids',
                                                   'words_lengths'],
                                      group_by_length=True,
                                      metric_for_best_model=args.metric_for_best_model,
                                      load_best_model_at_end=True,
                                      save_total_limit=2,
                                      #eval_steps=1,
                                      save_strategy=args.save_strategy,
                                      evaluation_strategy=args.eval_strategy,
                                      fp16=args.use_fp16,
                                      dataloader_num_workers=args.num_workers
                                      )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_loader.data_collator,
        compute_metrics=data_loader.compute_metrics
    )

    trainer.train()

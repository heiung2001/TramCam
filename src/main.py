import numpy as np
import os
import argparse

from transformers import Trainer, TrainingArguments
from transformers.models.auto import AutoTokenizer

from models.xlm_roberta import QuestionAnsweringWithXLMRoberta
from models.bert import QuestionAnsweringWithBert
from utils import data_loader
from datasets import load_metric

from peft.mapping import get_peft_model
from peft.tuners.lora import LoraConfig
from peft.utils.config import TaskType


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('-model',
                        default='QuestionAnsweringWithXLMRoberta',
                        choices=['QuestionAnsweringWithXLMRoberta',
                                 'QuestionAnsweringWithBert',
                                 'QuestionAnsweringWithPhoBert'],
                        type=str,
                        help='Choose a model to finetune.')

    # general
    parser.add_argument('-base_model', default='xlm-roberta-large', help='Model to fine-tune')
    parser.add_argument('-use_fp16', default=False, action='store_true', help="Whether to use fp16 or not")
    parser.add_argument('-save_strategy', default='epoch', choices=['epoch', 'steps'])
    parser.add_argument('-eval_strategy', default='epoch', choices=['epoch, steps'])
    parser.add_argument('-logging_steps', default=100, type=int, help='Logging after # steps')

    # hyperparameters
    parser.add_argument('-num_epochs', default=5, type=int, help="Number of epochs")
    parser.add_argument('-train_batch_size', default=8, type=int)
    parser.add_argument('-eval_batch_size', default=8, type=int)
    parser.add_argument('-metric_for_best_model', default='f1', choices=['f1', 'EM'])
    parser.add_argument('-gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('-num_workers', default=2, type=int)
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
    parser.add_argument('-use_lora', default=False, action='store_true', help='Whether to use lora or not')
    parser.add_argument('-lora_rank', default=1, type=int)
    parser.add_argument('-lora_dropout', default=0.1, type=float)
    parser.add_argument('-lora_alpha', default=16)
    parser.add_argument('-lora_bias', default='none')

    # debug
    parser.add_argument('-debug', default=False, action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    model = globals()[args.model].from_pretrained(
        pretrained_model_name_or_path=args.base_model,
        # local_files_only=True,
        cache_dir=args.pretrained_storage_path
    )

    if args.model == 'QuestionAnsweringWithXLMRoberta':
        tokenizer = AutoTokenizer.from_pretrained(
            'xlm-roberta-large',
            cache_dir=args.pretrained_storage_path)

    elif args.model == 'QuestionAnsweringWithBert':
        tokenizer = AutoTokenizer.from_pretrained(
            'bert-base-multilingual-uncased',
            cache_dir=args.pretrained_storage_path)
    
    else:
        raise ValueError("Tokenizer don't exist")

    print(model)
    print(model.config)

    peft_config = LoraConfig(
        task_type=TaskType.QUESTION_ANS,
        r=args.lora_rank,
        target_modules=["query", "key", "value"],
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias
    )
    
    if args.use_lora:
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    train_dataset, valid_dataset = data_loader.get_dataloader(
        train_path=args.train_path,
        valid_path=args.valid_path,
        debug=args.debug
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

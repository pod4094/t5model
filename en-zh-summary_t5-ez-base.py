# TRUE translator 

# make dataset
from datasets import load_dataset

# en_zh = load_dataset("en_zh")
# print(en_zh)

import pandas as pd

# en_zh.set_format(type="pandas")
# df = en_zh["train"][:]
# print(df.head())

# example_0 = list(df.columns)
# print(example_0)

# # change column
# example_0_df = pd.DataFrame({col:[value] for col, value in zip(('en','zh'), example_0)})
# df.columns = ('en', 'zh')

# en_zh_df = pd.concat([example_0_df, df],).reset_index(drop=True)
# print(en_zh_df.head())

# from datasets import Dataset
# dataset = Dataset.from_pandas(en_zh_df)
# print(dataset)

# num_train = 1200000
# num_valid = 90000
# num_test = 10000

# en_zh_df_train = en_zh_df.iloc[:num_train]
# en_zh_df_valid = en_zh_df.iloc[num_train:num_train+num_valid]
# en_zh_df_test = en_zh_df.iloc[-num_test:]

# en_zh_df_train.to_csv("train.tsv", sep='\t', index=False)
# en_zh_df_valid.to_csv("valid.tsv", sep='\t', index=False)
# en_zh_df_test.to_csv("test.tsv", sep='\t', index=False)

path = 'G:/dnn_work/Hand_On/spm-data/2024.6.2/완성/'
# path = '../datasets/tsv-0/'
data_files = {"train": path+"train.tsv", "valid": path+"valid.tsv", "test": path+"test.tsv"}
# dataset =  load_dataset("csv", data_files=data_files, delimiter="&&&")
dataset =  load_dataset("csv", data_files=data_files, delimiter="\t")

print(dataset)
print(dataset['train']['en'][0])

# Training
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, load_metric
import numpy as np
import torch
import multiprocessing

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
model_ckpt = "../models/ke-t5-base"
max_token_length = 64

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
print(dataset['train']['en'][10], dataset['train']['zh'][10])

tokenized_sample_en = tokenizer(dataset['train']['en'][10])
print(tokenized_sample_en)

tokenized_sample_zh = tokenizer(dataset['train']['zh'][10])
print(tokenized_sample_zh)

print(tokenizer(dataset['train']['en'][:3], padding=True))

print(pd.DataFrame(
    [
        tokenized_sample_en['input_ids'],
        tokenizer.convert_ids_to_tokens(tokenized_sample_en['input_ids'])
    ], index=('ids', 'tokens')
))

print(pd.DataFrame(
    [
        tokenized_sample_zh['input_ids'],
        tokenizer.convert_ids_to_tokens(tokenized_sample_zh['input_ids'])
    ], index=('ids', 'tokens')
))

def convert_examples_to_features(examples):
    ###########################################################################
    # # Setup the tokenizer for targets
    # # with, Older
    # input_encodings = tokenizer(examples['en'],
    #                             max_length=max_token_length, truncation=True)
    #
    # with tokenizer.as_target_tokenizer():
    #     target_encodings = tokenizer(text_target=examples['zh'],
    #                             max_length=max_token_length, truncation=True)
    #
    #
    #
    #
    # return {
    #     "input_ids": input_encodings["input_ids"],
    #     "attention_mask": input_encodings["attention_mask"],
    #     "labels": target_encodings["input_ids"]
    # }

    # Newer
    model_inputs = tokenizer(examples['en'],
                             text_target=examples['zh'],
                             max_length=max_token_length, truncation=True)

    return model_inputs

num_cpu = multiprocessing.cpu_count()
print(num_cpu)
tokenized_datasets = dataset.map(convert_examples_to_features, batched=True,
                                remove_columns=dataset['train'].column_names,
                                num_proc=1)

print(tokenized_datasets)

print(tokenized_datasets['train'][10])

print(dataset['train']['en'][10])
print(tokenized_datasets['train'][10]['input_ids'])
print(tokenizer.convert_ids_to_tokens(tokenized_datasets['train'][10]['input_ids']))

print(dataset['train']['zh'][10])
print(tokenized_datasets['train'][10]['labels'])
print(tokenizer.convert_ids_to_tokens(tokenized_datasets['train'][10]['labels']))

# Model
model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)
# print(model) # model.summary()???

# T5 has no zh, but how to add
encoder_inputs = tokenizer(
    ["Studies have been shown that owning a dog is good for you"],
    return_tensors="pt"
)['input_ids'].to(device)

decoder_targets = tokenizer(
    ["你好同学."],
    return_tensors="pt"
)['input_ids'].to(device)
print(encoder_inputs)
print(decoder_targets)

decoder_inputs = model._shift_right(decoder_targets)

print(
pd.DataFrame(
    [
        tokenizer.convert_ids_to_tokens(decoder_targets[0]),
        tokenizer.convert_ids_to_tokens(decoder_inputs[0])
    ],
    index=('decoder tgt', 'decoder input')
))

# forward passing
outputs = model(input_ids = encoder_inputs,
                decoder_input_ids = decoder_inputs,
                labels = decoder_targets)

print(outputs.keys())
print(outputs.loss)
print(outputs['encoder_last_hidden_state'].shape)
print(outputs['logits'].shape)
print(tokenizer.convert_ids_to_tokens( torch.argmax(outputs['logits'][0], axis=1).cpu().numpy() ))


# Collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

print(tokenized_datasets["train"][1:3])

print([tokenized_datasets["train"][i] for i in range(1, 3)])

batch = data_collator(
    [tokenized_datasets["train"][i] for i in range(1, 3)]
)
print(batch.keys())
print(batch)
#
# # Metric
# import evaluate
# metric = evaluate.load("sacrebleu")
#
# def compute_metrics(eval_preds):
#     preds, labels = eval_preds
#
#     if isinstance(preds, tuple):
#         preds = preds[0]
#
#     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
#
#     # Replace -100 in the labels as we can't decode them.
#     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
#
#     # Some simple post-processing
#     decoded_preds = [pred.strip() for pred in decoded_preds]
#     decoded_labels = [[label.strip()] for label in decoded_labels]
#
#     result = metric.compute(predictions=decoded_preds, references=decoded_labels)
#     result = {"bleu": result["score"]}
#
#     return result
# #
# # Trainer
# training_args = Seq2SeqTrainingArguments(
#     output_dir="en-zh-translator",
#     learning_rate=0.0005,
#     weight_decay=0.01,
#     per_device_train_batch_size=64,
#     per_device_eval_batch_size=128,
#     num_train_epochs=1,
#     save_steps=500,
#     save_total_limit=2,
#     evaluation_strategy="steps",
#     logging_strategy="no",
#     predict_with_generate=True,
#     fp16=True,
#     gradient_accumulation_steps=2,
#     report_to="none" # Wandb off
# )
#
# trainer = Seq2SeqTrainer(
#     model,
#     training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["valid"],
#     data_collator=data_collator,
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics,
# )
#
# trainer.train()
#
# trainer.save_model("en-zh-translator-kp")

#  # Test
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# model_dir = "./en-zh-translator-kp"
# # model_dir = "../models/ke-t5-base"
# tokenizer = AutoTokenizer.from_pretrained(model_dir)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
# # print(model.cuda())
# print(model.cpu())
# input_text = [
#     "Because deep learning frameworks are well developed, in these days, machine translation system can be built without anyone's help.",
#     "This system was made by using HuggingFace's T5 model for a one day",
#     "Kim Ok-gyun Japan's In a sabotage maneuver Of course "
# ]
# max_token_length = 64
# inputs = tokenizer(input_text, return_tensors="pt",
#                    padding=True, max_length=max_token_length)
# print(inputs)
# zhreans = model.generate(
#     **inputs,
#     max_length=max_token_length,
#     num_beams=5,
# )
#
# print(zhreans.shape)
# print(
# [
#     tokenizer.convert_tokens_to_string(
#     tokenizer.convert_ids_to_tokens(zhrean)) for zhrean in zhreans
# ])
#
# from torch.utils.data import DataLoader
# # translate test corpus's one iteration
# test_dataloader = DataLoader(
#     tokenized_datasets["test"], batch_size=32, collate_fn=data_collator
# )
# test_dataloader_iter = iter(test_dataloader)
# test_batch = next(test_dataloader_iter)
# print(test_batch.keys())
# test_input = { key: test_batch[key] for key in ('input_ids', 'attention_mask') }
# print(test_input.keys())
#
# zhreans = model.generate(
#     **test_input,
#     max_length=max_token_length,
#     num_beams=5,
# )
# print(zhreans)
#
# labels =  np.where(test_batch.labels != -100, test_batch.labels, tokenizer.pad_token_id)
# eng_sents = tokenizer.batch_decode(test_batch.input_ids, skip_special_tokens=True)[10:20]
# references = tokenizer.batch_decode(labels, skip_special_tokens=True)[10:20]
# preds = tokenizer.batch_decode( zhreans, skip_special_tokens=True )[10:20]
#
# for s in zip(eng_sents, references, preds):
#     print('English   :', s[0])
#     print('Reference :', s[1])
#     print('Translated:', s[2])
#     print('\n')

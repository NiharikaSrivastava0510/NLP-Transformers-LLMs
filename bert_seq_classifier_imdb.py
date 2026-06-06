# !/usr/bin/env python
# -*- coding: utf-8 -*-

######################################################################
#
# (c) Copyright University of Southampton, 2024
# Lab Steps 2 & 3: Fine-tune Yelp BERT on IMDB, evaluate macro F1
#
######################################################################

import os, sys, math
import numpy as np
from sklearn.metrics import f1_score

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.6,max_split_size_mb:256'
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import transformers
from datasets import load_dataset
import datasets
import evaluate

os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

def report_gpu_mem():
    t = torch.cuda.get_device_properties(0).total_memory / 1000000000
    r = torch.cuda.memory_reserved(0) / 1000000000
    a = torch.cuda.memory_allocated(0) / 1000000000
    print('cuda memory (free', math.ceil(r - a), '; reserved', math.ceil(r), '; allocated', math.ceil(a), ')')

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    report_gpu_mem()
    device = 'cuda'
else:
    print('cuda not available')
    device = 'cpu'

print('device used =', device)

###########################
# Step 2: Load IMDB dataset + pre-trained Yelp BERT model
# dataset  https://huggingface.co/datasets/stanfordnlp/imdb
# model    https://huggingface.co/textattack/bert-base-uncased-yelp-polarity
# IMDB is a binary sentiment classification task: positive (1) or negative (0)
###########################

imdb_split = load_dataset('imdb', split=['train', 'test'])
imdb = datasets.DatasetDict()
imdb['train'] = imdb_split[0]
imdb['test']  = imdb_split[1]
print('imdb dataset train size =', len(imdb['train']))
print('imdb dataset test size  =', len(imdb['test']))
print('imdb data point 0 =', imdb['train'][0])

# Use a subset to keep CPU training time manageable (~40 min)
# Full dataset: 25k train / 25k test (~8+ hrs on CPU)
train_data = imdb['train'].shuffle(seed=42).select(range(5000))
test_data  = imdb['test'].shuffle(seed=42).select(range(1000))
print('using subset: train =', len(train_data), ', test =', len(test_data))

###########################
# Step 2: Load pre-trained Yelp BERT model
###########################

model_name = 'textattack/bert-base-uncased-yelp-polarity'
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
model = model.to('cpu')

print('\nmodel architecture (Yelp BERT with sequence classification head):')
print(model)

###########################
# Step 2: Tokenize IMDB dataset
###########################

def preprocess_function(data_point):
    return tokenizer(data_point['text'], padding=True, truncation=True, return_tensors='pt')

encoded_sample = preprocess_function(test_data[:3])
print('encoded sample (batch of 3) =', encoded_sample)

encoded_train = train_data.map(preprocess_function, batched=True)
encoded_test  = test_data.map(preprocess_function, batched=True)
print('encoded train =', encoded_train)
print('encoded test  =', encoded_test)

###########################
# Step 3: Fine-tune + evaluate with macro F1
###########################

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    f1 = f1_score(labels, predictions, average='macro')
    return {'macro_f1': f1}

batch_size = 16
num_epochs = 5

args = transformers.TrainingArguments(
    output_dir='bert-yelp-finetuned-imdb',
    eval_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='macro_f1',
)

trainer = transformers.Trainer(
    model=model,
    args=args,
    train_dataset=encoded_train,
    eval_dataset=encoded_test,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

# Baseline: evaluate pre-trained Yelp model on IMDB before any fine-tuning
print('\nevaluating pre-trained yelp model on imdb (baseline, expected macro F1 ~0.87)')
baseline_results = trainer.evaluate()
print('baseline eval results =', baseline_results)

# Fine-tune on IMDB
print('\nfine-tuning yelp model on imdb dataset')
train_report = trainer.train()
print('training report =', train_report)

# Final evaluation of fine-tuned model (expected macro F1 ~0.92)
eval_results = trainer.evaluate()
print('eval results IMDB fine-tuned =', eval_results)

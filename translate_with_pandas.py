# -*- coding: utf-8 -*-

import os
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
import progressbar

def divide_chunks(l, n):
    # looping till length l 
    for i in range(0, len(l), n):
        yield l[i:i + n]

def flat_list(l):
    flat_list = [item for sublist in l for item in sublist]
    return flat_list

def translate(text_list, model=None, tokenizer=None):
    translated = model.generate(**tokenizer.prepare_translation_batch(text_list))
    tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    return tgt_text

model_name = 'Helsinki-NLP/opus-mt-es-ca'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)


file_train_src = "train.tsv"
file_test_src = 'test.tsv'

col_names = ['Label', 'Tweet']
df_train_src = pd.read_csv(file_train_src, sep='\t', names=col_names)
train_src = df_train_src.text.to_list()

df_test_src = pd.read_csv(file_test_src, sep='\t', names=col_names)
test_src = df_test_src.text.to_list()

train_src_lst = list(divide_chunks(train_src, 4))

train_src_translated_list = []

for i, chunk in enumerate(train_spa_lst):
    trans = translate(chunk, model=model, tokenizer=tokenizer)
    train_src_translated_list.append(trans)
    print('Chunks done: ', i)

train_src2tgt = flat_list(train_src_translated_list)

df_train_src['spa2cat'] = train_src2tgt
df_train_src = df_train_src[['labels', 'spa2cat']]
df_train_src.to_csv(file_train_tgt), sep='\t', header=None, index=False)

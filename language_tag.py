# 在文件头或代码中指定使用UTF-8编码
# -*- coding: utf-8 -*-
import os
import random
import re
import numpy as np
import torch
from sklearn.metrics import classification_report
from models import *



from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("../../../root/autodl-tmp/xlm-roberta-large")

def load_words():
    with open('words.txt') as word_file:
        valid_words = set(word_file.read().split())

    return valid_words
english_words = load_words()

def lang_tag(sentences1):
    lang_ids = []
    for sent in sentences1:
        array = tokenizer.tokenize(sent)
        lang_ids_t = [0]
        for i, element in enumerate(array):
                tmp = []
                if not element.startswith('▁'):
                    continue
                tmp.append(element[1:])

                for j in range(i+1,len(array)):
                    if  array[j].startswith('▁') :
                        break
                    else:
                        tmp.append(array[j])

                word = ''.join(tmp)
                word1 = word.lower()
                word2 = word.upper()

                if word1 in english_words:
                     lang = 1
                elif word2 in english_words:
                     lang = 1
                else:
                    lang = 2

                lang_ids_tmp = [lang] * (len(tmp))
                lang_ids_t = lang_ids_t + lang_ids_tmp
        lang_ids_t = lang_ids_t + [0]
        target_length = 80
        padding_length = max(0, target_length - len(lang_ids_t))
        lang_ids_t = lang_ids_t + [0] * padding_length
        lang_ids.append(lang_ids_t[:80])
    return lang_ids










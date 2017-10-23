import gensim
import os
import tflearn as tf
import numpy as np

dir = "F:/MEGA/Thesis/data/data/word2vec/"
filenames = ["train_negative_tokenized.txt","train_neutral_tokenized.txt","train_positive_tokenized.txt"]
stopWordFile = "./stopword_vi_small.txt"
sentences = []

with open(stopWordFile , "r", encoding='utf-8') as f:
    stopWords = f.read().split()

def pre_processline(line):
    line = [w.replace(u'\ufeff', '').lower() for w in line]
    line = [w for w in line if w not in stopWords]
    return line

for filename in filenames:
    path = os.path.join(dir, filename)
    f = open(path, 'r', encoding="utf-8")
    for line in f:
        line = line.split()
        if len(line) > 0:
            line = pre_processline(line)
            sentences.append(line)

w2vModel = gensim.models.Word2Vec(sentences, min_count=5, size=64, workers=4)
w2vModel.save("./w2vmodel")

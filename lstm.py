import gensim
import os
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
import numpy as np
maxLengthSequence = 20
w2vmodel = gensim.models.Word2Vec.load("./w2vmodel")
label = ['pos','neg','neu']
dirTrainCorpus = "F:/MEGA/Thesis/data/data/word2vec/"
trainCorpus = ["train_negative_tokenized.txt","train_neutral_tokenized.txt","train_positive_tokenized.txt"]
testCorpus = "test_tokenized_ANS.txt"

sentences = []
labels = []
num_dim = len(w2vmodel.wv.vocab)

def convert_data_to_index(string_data, wv):
    index_data = []
    for word in string_data:
        if word in wv:
            index_data.append(wv.vocab[word].index)
    return index_data


#Read train corpus
for i,fileTrain in enumerate(trainCorpus):
    path = os.path.join(dirTrainCorpus, fileTrain)
    f = open(path,'r',encoding='utf-8')
    for line in f:
        line = line.split()
        if len(line) >0:
            line = [w.replace(u'\ufeff', '').lower() for w in line]
            line = [w for w in line if w in w2vmodel.wv.vocab]
            sentences.append(line)
            labels.extend(label[i])

trainY = labels



# embedding matrix
embedding_matrix = np.zeros(len(w2vmodel.wv.vocab),64)
word_in_dict = []
for i in range(len(w2vmodel.wv.vocab)):
    embedding_vector = w2vmodel.wv[w2vmodel.wv.index2word[i]]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        word_in_dict.extend(w2vmodel.wv.index2word[i])




# Network building
net = tflearn.input_data([None, maxLengthSequence])
net = tflearn.embedding(net, input_dim=10000, output_dim=64)
net = tflearn.lstm(net, 64, dropout=0.8)
net = tflearn.fully_connected(net, 3, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
          batch_size=32)
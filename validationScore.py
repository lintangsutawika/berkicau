import re
import csv
import sys
import nltk
import gensim
import numpy as np
from sklearn.model_selection import KFold

import POStagger
from preprocess import findEntity

import Bi_LSTM
from Bi_LSTM import BiLSTM_CRF

from prepareData import prepareData 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

from evaluate import evaluate

modelName = sys.argv[1] 
mode = sys.argv[2] if len(sys.argv) > 2 else "None"
use = sys.argv[3] if len(sys.argv) > 3 else "TweetData" 
combine_embeddings = True if ((len(sys.argv) > 4) and (sys.argv[]=="True")) else False

nerData = prepareData()
if use=="10000Kalimat":
    if combine_embeddings==True:
        print("loading combinded embeddings, option: 1000Kalimat")
        word2vec = nerData.restore_model("./Embeddings/word2vec_with10000Kalimat_50Dimension.pic")
        tag2vec = nerData.restore_model("./Embeddings/tag2vec_with10000Kalimat_50Dimension.pic")
        dataPOSTag = nerData.restore_model("./Embeddings/tagFeed_with10000Kalimat.pic")
    else:
        print("loading word embeddings, option: 1000Kalimat")
        word2vec = nerData.restore_model("./Embeddings/word2vec_with10000Kalimat_100Dimension.pic")

elif use=="TweetData":
    if combine_embeddings==True:
        print("loading combinded embeddings, option: TweetData")
        word2vec = nerData.restore_model("./Embeddings/word2vec_withTweetData_50Dimension.pic")
        tag2vec = nerData.restore_model("./Embeddings/tag2vec_withTweetData_50Dimension.pic")
        dataPOSTag = nerData.restore_model("./Embeddings/tagFeed_withTweetData.pic")
    else:
        print("loading word embeddings, option: TweetData")
        word2vec = nerData.restore_model("./Embeddings/word2vec_withTweetData_100Dimension.pic")

elif use=="All":
    if combine_embeddings==True:
        print("loading combinded embeddings, option: All")
        word2vec = nerData.restore_model("./Embeddings/word2vec_All_50Dimension.pic")
        tag2vec = nerData.restore_model("./Embeddings/tag2vec_All_50Dimension.pic")
        dataPOSTag = nerData.restore_model("./Embeddings/tagFeed_All.pic")
    else:
        print("loading word embeddings, option: All")
        word2vec = nerData.restore_model("./Embeddings/word2vec_All_100Dimension.pic")

text = findEntity()
data, tags = text.corpus2BIO(mode=mode)
dataPOSTag = nerData.restore_model("./Embeddings/tagFeed_withTweetData.pic")

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 100
HIDDEN_DIM = 50
if mode == "withIntermediate":
    tag_to_ix = {"None": 0, "B-PER":1, "I-PER":2, "B-LOC":3, "I-LOC":4, "B-ORG":5, "I-ORG": 6, START_TAG:7, STOP_TAG:8} #Version 2
else:
    tag_to_ix = {"None": 0, "I-PER": 1, "I-LOC": 2, "I-ORG":3, START_TAG: 4, STOP_TAG: 5} #Version 1

model = torch.load(modelName+".pth") 

kf = KFold(n_splits=5)
for nFold, (train_index, test_index) in enumerate(kf.split(data)):
    if nFold == 0:        # optimizer = optim.Adadelta(model.parameters())
        X_train, X_test = np.asarray(data)[train_index], np.asarray(data)[test_index]
        y_train, y_test = np.asarray(tags)[train_index], np.asarray(tags)[test_index]
        if combine_embeddings == True:
            POS_train, POS_test = np.asarray(dataPOSTag)[train_index], np.asarray(dataPOSTag)[test_index]
        TrainIndex = train_index
        TestIndex = test_index
    else:
        break

#Print Predictions
allPredictions = []
for index,testTrain in enumerate(X_test):
    if combine_embeddings == True:
        embedded_sentence = torch.from_numpy(np.concatenate((np.asarray([word2vec.wv[w] for w in testTrain]), np.asarray([tag2vec.wv[p] for p in POS_test[index]])),axis=1))
    else:
        embedded_sentence = torch.from_numpy(np.asarray([word2vec.wv[w] for w in testTrain]))
    printSentence = ""
    for w in testTrain:
        printSentence = printSentence + " " + w

    targets = torch.LongTensor([t for t in y_test[index]])
    targetTags = targets.numpy()
    inputs, labels = autograd.Variable(embedded_sentence), autograd.Variable(targets)
    output = np.asarray(model(autograd.Variable(embedded_sentence))[1])
    allPredictions.append(output)

evaluation = evaluate()
outputText = evaluation.getProperOutput(allPredictions, Index=TestIndex, mode=mode)
yPred_Person, yPred_Location, yPred_Organization = evaluation.getPredictionElements(outputText)
yPerson, yLocation, yOrganization = evaluation.getTaggedElements(TestIndex)

F1_ExactMatch_Macro, F1_ExactMatch_Micro = evaluation.F1_ExactMatch(yPerson, yLocation, yOrganization, yPred_Person, yPred_Location, yPred_Organization)

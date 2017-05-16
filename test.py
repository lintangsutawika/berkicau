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

from entityTagger import entityTagger
from evaluate import evaluate

from lookupBase import lookUp

mode = "None"
use = "TweetData"#"All" #"1000Kalimat", #
combine_embeddings = False
code_name = "TEST"
use10000Kalimat = True
experiment = []
BATCHSIZE = 64.0

nerData = prepareData()
# if use=="10000Kalimat":
#     if combine_embeddings==True:
#         print("loading combinded embeddings, option: 1000Kalimat")
#         word2vec = nerData.restore_model("./Embeddings/word2vec_with10000Kalimat_50Dimension.pic")
#         tag2vec = nerData.restore_model("./Embeddings/tag2vec_with10000Kalimat_50Dimension.pic")
#         dataPOSTag = nerData.restore_model("./Embeddings/tagFeed_with10000Kalimat.pic")
#     else:
#         print("loading word embeddings, option: 1000Kalimat")
#         word2vec = nerData.restore_model("./Embeddings/word2vec_with10000Kalimat_100Dimension.pic")

# elif use=="TweetData":
#     if combine_embeddings==True:
#         print("loading combinded embeddings, option: TweetData")
#         word2vec = nerData.restore_model("./Embeddings/word2vec_withTweetData_50Dimension.pic")
#         tag2vec = nerData.restore_model("./Embeddings/tag2vec_withTweetData_50Dimension.pic")
#         dataPOSTag = nerData.restore_model("./Embeddings/tagFeed_withTweetData.pic")
#     else:
#         print("loading word embeddings, option: TweetData")
#         word2vec = nerData.restore_model("./Embeddings/word2vec_withTweetData_100Dimension.pic")

# elif use=="All":
#     if combine_embeddings==True:
#         print("loading combinded embeddings, option: All")
#         word2vec = nerData.restore_model("./Embeddings/word2vec_All_50Dimension.pic")
#         tag2vec = nerData.restore_model("./Embeddings/tag2vec_All_50Dimension.pic")
#         dataPOSTag = nerData.restore_model("./Embeddings/tagFeed_All.pic")
#     else:
#         print("loading word embeddings, option: All")
#         word2vec = nerData.restore_model("./Embeddings/word2vec_All_100Dimension.pic")

# toFeed = nerData.restore_model("./Embeddings/toFeed_withTweetData.pic")
# tagFeed = nerData.restore_model("./Embeddings/tagFeed_withTweetData.pic")
toFeed = nerData.getDefaultData()
toFeed = nerData.getTweetData(toFeed, filename="MoreTweets.tsv")
#Incorporate testCorpus to embeddings
testCorpus = findEntity(filename="test.txt")
tokenizedTest = testCorpus.corpus2BIO()[0]
for tokens in tokenizedTest:
    toFeed.append(tokens)

testTags = nerData.getTestTag(tokenizedTest)
# for element in testTags:
    # tagFeed.append(element)

if combine_embeddings == True:
    word2vec = nerData.getWord2Vec(toFeed)
    tag2vec = nerData.getTag2Vec(tagFeed)
else:
    word2vec = nerData.getWord2Vec(toFeed, dim=100)

entityTagger = entityTagger()
data = entityTagger.data
lookup = lookUp()
lookUpPredictions = lookup.getLookUpPredictions()

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 100
HIDDEN_DIM = 50
if mode == "withIntermediate":
    tag_to_ix = {"None": 0, "B-PER":1, "I-PER":2, "B-LOC":3, "I-LOC":4, "B-ORG":5, "I-ORG": 6, START_TAG:7, STOP_TAG:8} #Version 2
else:
    tag_to_ix = {"None": 0, "I-PER": 1, "I-LOC": 2, "I-ORG":3, START_TAG: 4, STOP_TAG: 5} #Version 1

modelFolder = re.split('\n',open("./Models/models3.txt", "rb").read())
models = []
#open model from a directory:
for modelName in modelFolder:
    models.append(torch.load("./Models/"+modelName+".pth"))

#Print Predictions
allPredictions = []
for index,testTrain in enumerate(tokenizedTest):
    finalPrediction = np.zeros(np.shape(testTrain), dtype=np.int32)
    if combine_embeddings == True:
        embedded_sentence = torch.from_numpy(np.concatenate((np.asarray([word2vec.wv[w] for w in testTrain]), np.asarray([tag2vec.wv[p] for p in testTags[index]])),axis=1))
    else:
        embedded_sentence = torch.from_numpy(np.asarray([word2vec.wv[w] for w in testTrain]))
    
    printSentence = ""
    for w in testTrain:
        printSentence = printSentence + " " + w

    print("Sentence:\n{}\n".format(printSentence))
    out = []
    for num,model in enumerate(models):
        out.append(model(autograd.Variable(embedded_sentence))[1])
        
    for num,pred in enumerate(out):
        print("Prediction {}:\n{}".format(num,np.asarray(pred)))

    out = np.asarray(out)
    for i in range(np.shape(out)[1]):
        count1 = np.count_nonzero(out[:,i]==1)
        count2 = np.count_nonzero(out[:,i]==2)
        count3 = np.count_nonzero(out[:,i]==3)
        if count1 == 0 and count2 == 0 and count3 == 0:
            finalPrediction[i]
        else:
            finalPrediction[i] = np.argmax([count1, count2, count3]) + 1
    
    print("finalPrediction:\n{}\nlookupPrediction:\n{}\n".format(finalPrediction, lookUpPredictions[index]))
    allPredictions.append(finalPrediction)



# evaluation = evaluate(filename="test.txt")
# text = evaluation.getProperOutput(allPredictions, [0,len(tokenizedTest)])
# with open("output_TweetData_CombinedFalse_NEWWORD2VEC.txt", 'w') as txtFile:
#     for lines in text:
#         txtFile.write(lines+"\r\n")




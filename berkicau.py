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

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from rujukKateglo import rujukKateglo

combine_embeddings = False

#External Arguments
testData = sys.argv[1]
experimentNum = int(sys.argv[2])
# verbosity = sys.argv[2]
#Assumes that the data is in the same directory as berkicau.py
directoryTestData = "./"

#Data preperation object
nerData = prepareData()
#TestData
testCorpus = findEntity(dir=directoryTestData,filename=testData)
tokenizedTest = testCorpus.corpus2BIO()[0]

if experimentNum == 1:
    combine_embeddings = True
    print("Experiment 1: Word2Vec + Tag2Vec with TweetData")
    modelFolder = re.split('\n',open("./Models/models_TweetData_Combined.txt", "rb").read())
    toFeed = nerData.restore_model("./Embeddings/toFeed_withTweetData.pic")
    tagFeed = nerData.restore_model("./Embeddings/tagFeed_withTweetData.pic")
    #Incorporate testCorpus to embeddings
    for tokens in tokenizedTest:
        toFeed.append(tokens)

    if combine_embeddings == True:
        word2vec = nerData.getWord2Vec(toFeed)
        tag2vec = nerData.getTag2Vec(tagFeed)
        testTags = nerData.getTestTag(tokenizedTest)

elif experimentNum == 2:
    print("Experiment 2: Word2Vec with TweetData")
    modelFolder = re.split('\n',open("./Models/models_TweetData.txt", "rb").read())
    toFeed = nerData.restore_model("./Embeddings/toFeed_withTweetData.pic")
     #Incorporate testCorpus to embeddings
    for tokens in tokenizedTest:
        toFeed.append(tokens)

    word2vec = nerData.getWord2Vec(toFeed, dim=100)

elif experimentNum == 3:
    print("Experiment 3: Lookup table")
    print("Getting prediction from lookup")
    lookup = lookUp(testfile=directoryTestData+testData)
    finalOutput = lookup.getLookUpPredictions()

elif experimentNum == 4:
    print("Experiment 4: Lookup table with self added datasets")
    print("Getting prediction from lookup")
    lookup = lookUp(testfile=directoryTestData+testData, withExtra=True)
    finalOutput = lookup.getLookUpPredictions()

elif experimentNum in [5,6,7,8,9,10]:
    
    if experimentNum == 5:
        print("Experiment 5: Word2Vec from larger TweetData")
    elif experimentNum == 6:
        print("Experiment 6: Word2Vec from larger TweetData with Lookup Table")
    elif experimentNum == 7:
        print("Experiment 7: Word2Vec from larger TweetData with Lookup Table and Stemmer+Kateglo Checker")    
    elif experimentNum == 8:
       print("Experiment 8: Word2Vec from larger TweetData with Lookup Table and Stemmer+Kateglo Checker Ensemble Vote all Class")
    elif experimentNum == 9:
       print("Experiment 9: Word2Vec from larger TweetData with Weighted Lookup Table and Stemmer+Kateglo Checker Ensemble Vote all Class")
    elif experimentNum == 10:
       print("Experiment 9: Word2Vec from larger TweetData with Weighted Extra Lookup Table and Stemmer+Kateglo Checker Ensemble Vote all Class (8)")

    if experimentNum in [6,7,8,9]:
        lookup = lookUp(testfile=directoryTestData+testData)
        print("Getting prediction from lookup")
        lookUpPredictions = lookup.getLookUpPredictions()

    if experimentNum == 10:
        lookup = lookUp(testfile=directoryTestData+testData,withExtra=True)
        print("Getting prediction from lookup")
        lookUpPredictions = lookup.getLookUpPredictions()

    print("\nUpdating Word2Vec with entry sentences")
    word2vec = nerData.restore_model("./Embeddings/word2vec_withTweetData2_withTest_100Dimension.pic")
    word2vec.build_vocab(tokenizedTest, update=True)
    word2vec.train(tokenizedTest,total_examples=len(tokenizedTest),epochs=word2vec.iter)
    
    modelFolder = re.split('\n',open("./Models/model_TweetData2.txt", "rb").read())
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    rujukan = rujukKateglo()



if experimentNum in [1,2,5,6,7,8,9,10]:
    models = []
    #open model from a directory:
    for modelName in modelFolder:
        models.append(torch.load("./Models/"+modelName+".pth"))
    
    print("Outputing predictions")
    #Print Predictions
    allPredictions = []
    finalOutput = []
    for index,testEntry in enumerate(tokenizedTest):
        finalPrediction = np.zeros(np.shape(testEntry), dtype=np.int32)
        if combine_embeddings == True:
            embedded_sentence = torch.from_numpy(np.concatenate((np.asarray([word2vec.wv[w] for w in testEntry]), np.asarray([tag2vec.wv[p] for p in testTags[index]])),axis=1))
        else:
            embedded_sentence = torch.from_numpy(np.asarray([word2vec.wv[w] for w in testEntry]))
        
        printSentence = ""
        for w in testEntry:
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
            if experimentNum in [8,9]:
                count0 = 10 - (count1+count2+count3)
            elif experimentNum == 10:
                count0 = 8 - (count1+count2+count3)
            
            if experimentNum in [6,7,8,9,10]:
                if lookUpPredictions[index][i] == 1:
                    if experimentNum == 9:
                        count1 += 4
                    elif experimentNum == 10:
                        count1 += 5
                elif lookUpPredictions[index][i] == 2:
                    if experimentNum == 9:
                        count2 += 4
                    elif experimentNum == 10:
                        count1 += 5
                elif lookUpPredictions[index][i] == 3:
                    if experimentNum == 9:
                        count3 += 4
                    elif experimentNum == 10:
                        count1 += 5
            
            if count1 == 0 and count2 == 0 and count3 == 0:
                finalPrediction[i] = 0
            else:
                if experimentNum in [8,9,10]:
                    finalPrediction[i] = np.argmax([count0, count1, count2, count3])
                else:
                    finalPrediction[i] = np.argmax([count1, count2, count3]) + 1

        tempOut = np.zeros(np.shape(finalPrediction), dtype=np.int32)
        for indWord,element in enumerate(finalPrediction):
            if experimentNum in [6,7]:
                if lookUpPredictions[index][indWord] != 0:
                    tempOut[indWord] = lookUpPredictions[index][indWord]
                elif element != 0:
                    # print("{}:{}".format(tokenizedTest[index][indWord], rujukan.jenis_kata(stemmer.stem(tokenizedTest[index][indWord]))))
                    if experimentNum == 7:
                        wordType=rujukan.jenis_kata(stemmer.stem(tokenizedTest[index][indWord]))
                        if re.match("^[A-Za-z0-9.]*$",tokenizedTest[index][indWord]) and(wordType=="NOTFOUND" or wordType=="NN" or wordType=="KP"):
                            tempOut[indWord] = element
                    else:
                        if re.match("^[A-Za-z0-9.]*$",tokenizedTest[index][indWord]):
                            tempOut[indWord] = element

            elif experimentNum in [8,9,10]:
                if element != 0:
                    # print("{}:{}".format(tokenizedTest[index][indWord], rujukan.jenis_kata(stemmer.stem(tokenizedTest[index][indWord]))))
                    wordType=rujukan.jenis_kata(stemmer.stem(tokenizedTest[index][indWord]))
                    if re.match("^[A-Za-z0-9.&]*$",tokenizedTest[index][indWord]) and(wordType=="NOTFOUND" or wordType=="NN" or wordType=="KP"):
                        tempOut[indWord] = element
            else:
                if re.match("^[A-Za-z0-9.&]*$",tokenizedTest[index][indWord]):
                    pass
                else:
                    finalPrediction[indWord] = 0

        if experimentNum in [6,7,8,9,10]:
            print("finalPrediction:\n{}\nlookupPrediction:\n{}\nfinalOutput:\n{}\n".format(finalPrediction, lookUpPredictions[index],tempOut))
        else:
            tempOut = finalPrediction
            print("finalOutput:\n{}\n".format(finalPrediction))

        finalOutput.append(tempOut)

evaluation = evaluate(dir=directoryTestData, filename=testData)
text = evaluation.getTextOutput(finalOutput, [0,len(tokenizedTest)])
print("Writing Output")
with open("output_Experiment_{}.txt".format(experimentNum), 'w') as txtFile:
    for lines in text:
        txtFile.write(lines+"\r\n")




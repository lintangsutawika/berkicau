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

class entityTagger(object):

    def __init__(self,mode = "None", use = "TweetData", combine_embeddings=False, BATCHSIZE=64.0):
        #use = "1000Kalimat", "TweetData"        
        self.mode = mode
        self.use = use
        self.combine_embeddings = combine_embeddings
        self.BATCHSIZE = BATCHSIZE
        self.EMBEDDING_DIM = 100
        self.HIDDEN_DIM = 50

        self.nerData = prepareData()
        self.evaluation = evaluate()
        self.text = findEntity()
        self.data, self.tags = self.text.corpus2BIO(mode=self.mode)

        START_TAG = "<START>"
        STOP_TAG = "<STOP>"
        if self.mode == "withIntermediate":
            self.tag_to_ix = {"None": 0, "B-PER":1, "I-PER":2, "B-LOC":3, "I-LOC":4, "B-ORG":5, "I-ORG": 6, START_TAG:7, STOP_TAG:8} #Version 2
        else:
            self.tag_to_ix = {"None": 0, "I-PER": 1, "I-LOC": 2, "I-ORG":3, START_TAG: 4, STOP_TAG: 5} #Version 1

        if self.use=="10000Kalimat":
            if self.combine_embeddings==True:
                print("loading combinded embeddings, option: 1000Kalimat")
                self.word2vec = self.nerData.restore_model("./Embeddings/word2vec_with10000Kalimat_50Dimension.pic")
                self.tag2vec = self.nerData.restore_model("./Embeddings/tag2vec_with10000Kalimat_50Dimension.pic")
                self.dataPOSTag = self.nerData.restore_model("./Embeddings/tagFeed_with10000Kalimat.pic")
            else:
                print("loading word embeddings, option: 1000Kalimat")
                self.word2vec = self.nerData.restore_model("./Embeddings/word2vec_with10000Kalimat_100Dimension.pic")

        elif self.use=="TweetData":
            if self.combine_embeddings==True:
                print("loading combinded embeddings, option: TweetData")
                self.word2vec = self.nerData.restore_model("./Embeddings/word2vec_withTweetData_50Dimension.pic")
                self.tag2vec = self.nerData.restore_model("./Embeddings/tag2vec_withTweetData_50Dimension.pic")
                self.dataPOSTag = self.nerData.restore_model("./Embeddings/tagFeed_withTweetData.pic")
            else:
                print("loading word embeddings2, option: TweetData")
                self.word2vec = self.nerData.restore_model("./Embeddings/word2vec_withTweetData2_100Dimension.pic")

        elif self.use=="All":
            if self.combine_embeddings==True:
                print("loading combinded embeddings, option: All")
                self.word2vec = self.nerData.restore_model("./Embeddings/word2vec_All_50Dimension.pic")
                self.tag2vec = self.nerData.restore_model("./Embeddings/tag2vec_All_50Dimension.pic")
                self.dataPOSTag = self.nerData.restore_model("./Embeddings/tagFeed_All.pic")
            else:
                print("loading word embeddings, option: All")
                self.word2vec = self.nerData.restore_model("./Embeddings/word2vec_All_100Dimension.pic")

    def printProgressBar (self, iteration, total, prefix = '', suffix = '',decimals = 1, length = 100, fill = '#'):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print'\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix),
        sys.stdout.flush()                                                                                                                                                                                                             
        if iteration == total:
            print()


    def trainModel(self, nFold, train_index, test_index, embeddings, trainedModel=None, num_epochs=100, model_name=None):
        if self.combine_embeddings == True:
            word2vec, tag2vec = embeddings
        else:
            word2vec = embeddings
        if trainedModel == None:
            model = BiLSTM_CRF(self.tag_to_ix, self.EMBEDDING_DIM, self.HIDDEN_DIM)
        else:
            model = trainedModel
            print("using trainedModel")
        # optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-4, momentum=0.9, nesterov=True)
        optimizer = optim.Adagrad(model.parameters(), lr=0.1, weight_decay=1e-4)
        # optimizer = optim.Adadelta(model.parameters())
        X_train, X_test = np.asarray(self.data)[train_index], np.asarray(self.data)[test_index]
        y_train, y_test = np.asarray(self.tags)[train_index], np.asarray(self.tags)[test_index]

        if self.combine_embeddings == True:
            POS_train, POS_test = np.asarray(self.dataPOSTag)[train_index], np.asarray(self.dataPOSTag)[test_index]

        recordLoss = 0
        LastF1_Micro = 0
        LastF1_Macro = 0
        bestModel_Macro = model
        bestModel_Micro = model
        for epoch in range(num_epochs):
            batchX = np.array_split(X_train, 384/self.BATCHSIZE)
            batchY = np.array_split(y_train, 384/self.BATCHSIZE)
            
            if self.combine_embeddings == True:
                batchPOS = np.array_split(POS_train, 384/self.BATCHSIZE)
            
            epochLoss = 0
            for batchInd,batchElement in enumerate(batchX):
                totalLoss = 0
                for index,sentenceTrain in enumerate(batchElement):
                    model.zero_grad()

                    if self.combine_embeddings == True:
                        embedded_sentence = torch.from_numpy(np.concatenate((np.asarray([word2vec.wv[w] for w in sentenceTrain]), np.asarray([tag2vec.wv[p] for p in batchPOS[batchInd][index]])),axis=1))
                    else:
                        embedded_sentence = torch.from_numpy(np.asarray([word2vec.wv[w] for w in sentenceTrain]))
                    
                    targets = torch.LongTensor([t for t in batchY[batchInd][index]])

                    inputs, labels = autograd.Variable(embedded_sentence), autograd.Variable(targets)

                    neg_log_likelihood = model.neg_log_likelihood(inputs, targets)
                    print(neg_log_likelihood.data.numpy()[0])
                    totalLoss += neg_log_likelihood
                    Log = "nFold: {}, Epoch: {}, Batch: {}\n".format(nFold, epoch, batchInd)
                    print(Log)
                    self.printProgressBar (index, self.BATCHSIZE-1)

                epochLoss += totalLoss
                gradientLoss = totalLoss/16.0  #BATCHSIZE
                gradientLoss.backward()
                optimizer.step()

            #EVALUATION
            valLoss = 0

            #Print Predictions
            allPredictions = []
            for index,testTrain in enumerate(X_test):
                
                if self.combine_embeddings == True:
                    embedded_sentence = torch.from_numpy(np.concatenate((np.asarray([word2vec.wv[w] for w in testTrain]), np.asarray([tag2vec.wv[p] for p in POS_test[index]])),axis=1))
                else:
                    embedded_sentence = torch.from_numpy(np.asarray([word2vec.wv[w] for w in testTrain]))
                printSentence = ""
                for w in testTrain:
                    printSentence = printSentence + " " + w

                targets = torch.LongTensor([t for t in y_test[index]])
                targetTags = targets.numpy()
                inputs, labels = autograd.Variable(embedded_sentence), autograd.Variable(targets)
                valLoss += model.neg_log_likelihood(inputs, targets)
                output = np.asarray(model(autograd.Variable(embedded_sentence))[1])
                allPredictions.append(output)
                print("Sentence:\n{}\nTag:\n{}\nPredict:\n{}".format(printSentence, targets.numpy(), output)) 

            trainLoss = epochLoss/np.shape(X_train)[0]
            validationLoss = valLoss/np.shape(X_test)[0]
            validationLoss = validationLoss.data.numpy()

            print("Epoch: {}, TrainLoss: {}, ValidationLoss: {}".format(epoch, trainLoss.data.numpy(), validationLoss))
            with open("performance_model_MODE{}_USE{}2_COMBINE{}_FOLD{}.csv".format(self.mode, self.use, self.combine_embeddings, nFold), 'a') as csvFile:
                writer = csv.writer(csvFile)
                # writer.writerow(["Epoch","TrainLoss","ValidationLoss","None","B-LOC","I-LOC","B-ORG","I-ORG","B-PER","I-PER"])
                writer.writerow([nFold, epoch, trainLoss.data.numpy()[0], validationLoss[0]]) #"None","B-LOC","I-LOC","B-ORG","I-ORG","B-PER","I-PER"])
                csvFile.close()

            outputText = self.evaluation.getProperOutput(allPredictions, Index=test_index, mode=self.mode)
            yPred_Person, yPred_Location, yPred_Organization = self.evaluation.getPredictionElements(outputText)
            yPerson, yLocation, yOrganization = self.evaluation.getTaggedElements(test_index)

            F1_ExactMatch_Macro, F1_ExactMatch_Micro = self.evaluation.F1_ExactMatch(yPerson, yLocation, yOrganization, yPred_Person, yPred_Location, yPred_Organization)

            if F1_ExactMatch_Micro > LastF1_Micro:
                if model_name==None:
                    torch.save(model, "model_MODE{}_USE{}2_COMBINE{}_FOLD{}_EPOCH{}_MICRO_EXACT.pth".format(self.mode, self.use, self.combine_embeddings, nFold, epoch))
                else:
                    torch.save(model,model_name)
                print("F1 Score: {}".format(F1_ExactMatch_Micro))
                LastF1_Micro = F1_ExactMatch_Micro
                bestModel_Micro = model   

            if F1_ExactMatch_Macro > LastF1_Macro:
                if model_name==None:
                    torch.save(model, "model_MODE{}_USE{}2_COMBINE{}_FOLD{}_EPOCH{}_MACRO_EXACT.pth".format(self.mode, self.use, self.combine_embeddings, nFold, epoch))
                else:
                    torch.save(model,model_name)
                print("F1 Score: {}".format(F1_ExactMatch_Macro))
                LastF1_Macro = F1_ExactMatch_Macro
                bestModel_Macro = model

        return bestModel_Macro, bestModel_Micro

if __name__ == '__main__':
    entityTagger = entityTagger()
    data = entityTagger.data

    kf = KFold(n_splits=5)
    for nFold, (train_index, test_index) in enumerate(kf.split(data)):
        entityTagger.trainModel(nFold, train_index, test_index, (entityTagger.word2vec))

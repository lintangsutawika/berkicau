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


mode = "None"
use = "All" #"TweetData"# #"1000Kalimat", 
combine_embeddings = True
code_name = "TEST"
use10000Kalimat = True
experiment = []
BATCHSIZE = 64.0

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

# nerData.restore_model("toFeed_with10000Kalimat.pic")
# nerData.restore_model("tagFeed_with10000Kalimat.pic")
# nerData.restore_model("toFeed_withTweetData.pic")
# nerData.restore_model("tagFeed_withTweetData.pic")
# nerData.restore_model("toFeed_All.pic")
# nerData.restore_model("tagFeed_All.pic")

text = findEntity()
data, tags = text.corpus2BIO(mode=mode)

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 100
HIDDEN_DIM = 50
if mode == "withIntermediate":
    tag_to_ix = {"None": 0, "B-PER":1, "I-PER":2, "B-LOC":3, "I-LOC":4, "B-ORG":5, "I-ORG": 6, START_TAG:7, STOP_TAG:8} #Version 2
else:
    tag_to_ix = {"None": 0, "I-PER": 1, "I-LOC": 2, "I-ORG":3, START_TAG: 4, STOP_TAG: 5} #Version 1


def printProgressBar (iteration, total, prefix = '', suffix = '',decimals = 1, length = 100, fill = '#'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print'\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix),
    sys.stdout.flush()                                                                                                                                                                                                             
    if iteration == total:
        print()

if __name__ == '__main__':
    
    kf = KFold(n_splits=5)
    for nFold, (train_index, test_index) in enumerate(kf.split(data)):

        model = BiLSTM_CRF(100, tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
        # optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-4, momentum=0.9, nesterov=True)
        optimizer = optim.Adagrad(model.parameters(), lr=0.1, weight_decay=1e-4)
        # optimizer = optim.Adadelta(model.parameters())
        X_train, X_test = np.asarray(data)[train_index], np.asarray(data)[test_index]
        y_train, y_test = np.asarray(tags)[train_index], np.asarray(tags)[test_index]

        if combine_embeddings == True:
            POS_train, POS_test = np.asarray(dataPOSTag)[train_index], np.asarray(dataPOSTag)[test_index]

        pastValLoss = np.asarray([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        recordLoss = 0
        LastF1_Micro = 0
        LastF1_Macro = 0
        for epoch in range(100):
            batchX = np.array_split(X_train, 384/BATCHSIZE)
            batchY = np.array_split(y_train, 384/BATCHSIZE)
            
            if combine_embeddings == True:
                batchPOS = np.array_split(POS_train, 384/BATCHSIZE)
            
            epochLoss = 0
            for batchInd,batchElement in enumerate(batchX):
                totalLoss = 0
                for index,sentenceTrain in enumerate(batchElement):
                    model.zero_grad()

                    if combine_embeddings == True:
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
                    printProgressBar (index, BATCHSIZE-1)

                epochLoss += totalLoss
                gradientLoss = totalLoss/16.0  #BATCHSIZE
                gradientLoss.backward()
                optimizer.step()

            #EVALUATION
            if mode=="withIntermediate":
                confusionMatrix = np.zeros([9,9])
            else:
                confusionMatrix = np.zeros([4,4])
            valLoss = 0

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
                valLoss += model.neg_log_likelihood(inputs, targets)
                output = np.asarray(model(autograd.Variable(embedded_sentence))[1])
                allPredictions.append(output)
                for indices, tag in enumerate(targetTags):
                    confusionMatrix[tag, output[indices]] += 1 
                
                print("Sentence:\n{}\nTag:\n{}\nPredict:\n{}".format(printSentence, targets.numpy(), output))

            trainLoss = epochLoss/np.shape(X_train)[0]
            validationLoss = valLoss/np.shape(X_test)[0]
            validationLoss = validationLoss.data.numpy()

            print("Epoch: {}, TrainLoss: {}, ValidationLoss: {}".format(epoch, trainLoss.data.numpy(), validationLoss))
            with open("performance_model_MODE{}_USE{}_COMBINE{}_FOLD{}.csv".format(mode, use, combine_embeddings, nFold), 'a') as csvFile:
                writer = csv.writer(csvFile)
                # writer.writerow(["Epoch","TrainLoss","ValidationLoss","None","B-LOC","I-LOC","B-ORG","I-ORG","B-PER","I-PER"])
                writer.writerow([nFold, epoch, trainLoss.data.numpy()[0], validationLoss[0]]) #"None","B-LOC","I-LOC","B-ORG","I-ORG","B-PER","I-PER"])
                csvFile.close()

            originalTokens = text.getTokenized()[test_index[0]:(test_index[-1]+1)]
            outputText = []
            for sentenceIndex,eachSentence in enumerate(allPredictions):
                outputSentence = ""
                for tagIndex, eachTag in enumerate(eachSentence):
                    predictedEntity = ""
                    if mode == "withIntermediate":
                        if eachTag == 0:
                            pass
                        elif eachTag == 1:
                            pass
                        elif eachTag == 3:
                            pass
                        elif eachTag == 5:
                            pass
                    else:
                        if eachTag == 0:
                            predictedEntity = originalTokens[sentenceIndex][tagIndex]
                        elif eachTag == 1:
                            if tagIndex == 0 or eachSentence[tagIndex - 1] != 1:
                                predictedEntity = "<ENAMEX TYPE=\"PERSON\">" + originalTokens[sentenceIndex][tagIndex]
                                if tagIndex == (len(eachSentence)-1) or eachSentence[tagIndex + 1] != 1:
                                    predictedEntity = predictedEntity + "</ENAMEX>"
                            elif tagIndex == (len(eachSentence)-1) or eachSentence[tagIndex + 1] != 1:
                                predictedEntity = originalTokens[sentenceIndex][tagIndex] + "</ENAMEX>"    
                            elif eachSentence[tagIndex - 1] == 1 and eachSentence[tagIndex + 1] == 1:
                                predictedEntity = originalTokens[sentenceIndex][tagIndex]
                            
                        elif eachTag == 2:
                            if tagIndex == 0 or eachSentence[tagIndex - 1] != 2:
                                predictedEntity = "<ENAMEX TYPE=\"LOCATION\">" + originalTokens[sentenceIndex][tagIndex]
                                if tagIndex == (len(eachSentence)-1) or eachSentence[tagIndex + 1] != 2:
                                    predictedEntity = predictedEntity + "</ENAMEX>"
                            elif tagIndex == (len(eachSentence)-1) or eachSentence[tagIndex + 1] != 2:
                                predictedEntity = originalTokens[sentenceIndex][tagIndex] + "</ENAMEX>"  
                            elif eachSentence[tagIndex - 1] == 2 and eachSentence[tagIndex + 1] == 2:
                                predictedEntity = originalTokens[sentenceIndex][tagIndex]

                        elif eachTag == 3:
                            if tagIndex == 0 or eachSentence[tagIndex - 1] != 3:
                                predictedEntity = "<ENAMEX TYPE=\"ORGANIZATION\">" + originalTokens[sentenceIndex][tagIndex]
                                if tagIndex == (len(eachSentence)-1) or eachSentence[tagIndex + 1] != 3:
                                    predictedEntity = predictedEntity + "</ENAMEX>"
                            elif tagIndex == (len(eachSentence)-1) or eachSentence[tagIndex + 1] != 3:
                                predictedEntity = originalTokens[sentenceIndex][tagIndex] + "</ENAMEX>"  
                            elif eachSentence[tagIndex - 1] == 3 and eachSentence[tagIndex - 1] == 3:
                                predictedEntity = originalTokens[sentenceIndex][tagIndex] 
                        
                    outputSentence = outputSentence + " " + predictedEntity
                outputText.append(outputSentence)

            predicted_corpus = outputText
            yPred_Person, yPred_Location, yPred_Organization = text.find_enamex(corpus=predicted_corpus)

            test_corpus = text.corpus[test_index[0]:(test_index[-1]+1)]
            yPerson, yLocation, yOrganization = text.find_enamex(corpus=test_corpus)

            true_positive = np.zeros(3)
            false_positive = np.zeros(3)
            false_negative = np.zeros(3)

            precision = np.zeros(3)
            recall = np.zeros(3)


            for predictions in yPred_Person:
                if (predictions in yPerson) == True:
                    true_positive[0] += 1.0
                else:
                    false_positive[0] +=1.0

            for predictions in yPred_Location:
                if (predictions in yLocation) == True:
                    true_positive[1] += 1.0
                else:
                    false_positive[1] +=1.0

            for predictions in yPred_Organization:
                if (predictions in yOrganization) == True:
                    true_positive[2] += 1.0
                else:
                    false_positive[2] +=1.0

            false_negative[0] = len(yPerson) - true_positive[0]
            false_negative[1] = len(yLocation) - true_positive[1]
            false_negative[2] = len(yOrganization) - true_positive[2]   

            #Macro Average
            for i in range(3):
                if np.sum(true_positive[i]) == 0:
                    precision[i] = 0
                    recall[i] = 0
                else:
                    precision[i] = true_positive[i] / (true_positive[i] + false_positive[i])
                    recall[i] = true_positive[i] / (true_positive[i] + false_negative[i])

            MacroPrecision = np.average(precision)
            MacroRecall = np.average(recall)
            if MacroPrecision == 0 or MacroRecall == 0:
                F1_ExactMatch_Macro = 0
            else:
                F1_ExactMatch_Macro = 2.0 * (MacroPrecision * MacroRecall)/(MacroPrecision + MacroRecall)

            #Micro Average
            if np.sum(true_positive) == 0:
                F1_ExactMatch_Micro = 0
            else:
                MicroPrecision = np.sum(true_positive)/(np.sum(true_positive)+np.sum(false_positive))
                MicroRecall = np.sum(true_positive)/(np.sum(true_positive)+np.sum(false_negative))
                F1_ExactMatch_Micro = 2.0 * (MicroPrecision * MicroRecall)/(MicroPrecision + MicroRecall)

            if F1_ExactMatch_Micro > LastF1_Micro:
                torch.save(model, "model_MODE{}_USE{}_COMBINE{}_FOLD{}_EPOCH{}_MICRO_EXACT.pth".format(mode, use, combine_embeddings, nFold, epoch))
                print("F1 Score: {}".format(F1_ExactMatch_Micro))
                LastF1_Micro = F1_ExactMatch_Micro   

            if F1_ExactMatch_Macro > LastF1_Macro:
                torch.save(model, "model_MODE{}_USE{}_COMBINE{}_FOLD{}_EPOCH{}_MACRO_EXACT.pth".format(mode, use, combine_embeddings, nFold, epoch))
                print("F1 Score: {}".format(F1_ExactMatch_Macro))
                LastF1_Macro = F1_ExactMatch_Macro

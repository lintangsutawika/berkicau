import re
import csv
import sys
import nltk
import gensim
import string
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

class evaluate(object):
    """docstring for evaluate"""
    def __init__(self,filename=None):
        super(evaluate, self).__init__()
        if filename==None:
            self.text = findEntity()
        else:
            self.text = findEntity(filename=filename)

    def getProperOutput(self, allPredictions, Index, mode="None"):
        originalTokens = self.text.getTokenized()[Index[0]:(Index[-1]+1)]
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
        return outputText

    def getTextOutput(self, allPredictions, Index, mode="None"):
        punctuations = list(string.punctuation)
        originalTokens = self.text.getTokenized()
        originalText = self.text.corpus

        for sentenceIndex,eachSentence in enumerate(allPredictions):
            tokenIndex = 0
            for tagIndex, eachTag in enumerate(eachSentence):
                currentToken = originalTokens[sentenceIndex][tagIndex]
                # if re.match("^[A-Za-z0-9]*$",currentToken):
                #     pass
                # else:
                #     eachTag = 0
                #     eachSentence[tagIndex] = 0

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
                    if eachTag == 1:
                        if tagIndex == 0 or eachSentence[tagIndex-1] != 1:
                            currentIndex = originalText[sentenceIndex][tokenIndex::].index(currentToken)
                            originalText[sentenceIndex] = originalText[sentenceIndex][0:tokenIndex] + originalText[sentenceIndex][tokenIndex:(tokenIndex+currentIndex)] + "<ENAMEX TYPE=\"PERSON\">" + originalText[sentenceIndex][(tokenIndex+currentIndex)::]
                            
                            if (tagIndex != (len(eachSentence)-1) and eachSentence[tagIndex + 1] != 1) or tagIndex == (len(eachSentence)-1):
                                currentIndex = originalText[sentenceIndex][tokenIndex::].index(currentToken)
                                originalText[sentenceIndex] = originalText[sentenceIndex][0:tokenIndex] + originalText[sentenceIndex][tokenIndex:(tokenIndex+currentIndex+len(currentToken))] + "</ENAMEX>" + originalText[sentenceIndex][(tokenIndex+currentIndex+len(currentToken))::]
                            currentIndex = originalText[sentenceIndex].index(currentToken)
                            tokenIndex = currentIndex + len(currentToken)
                        elif tagIndex != (len(eachSentence)-1) and eachSentence[tagIndex - 1] == 1 and eachSentence[tagIndex + 1] == 1:
                            pass
                        elif (tagIndex != (len(eachSentence)-1) and eachSentence[tagIndex + 1] != 1):
                            currentIndex = originalText[sentenceIndex][tokenIndex::].index(currentToken)
                            originalText[sentenceIndex] = originalText[sentenceIndex][0:tokenIndex] + originalText[sentenceIndex][tokenIndex:(tokenIndex+currentIndex+len(currentToken))] + "</ENAMEX>" + originalText[sentenceIndex][(tokenIndex+currentIndex+len(currentToken))::]
                            currentIndex = originalText[sentenceIndex].index(currentToken)
                            tokenIndex = currentIndex + len(currentToken)
                        elif tagIndex == (len(eachSentence)-1):
                            currentIndex = originalText[sentenceIndex][tokenIndex::].index(currentToken)
                            originalText[sentenceIndex] = originalText[sentenceIndex][0:tokenIndex] + originalText[sentenceIndex][tokenIndex:(tokenIndex+currentIndex+len(currentToken))] + "</ENAMEX>" + originalText[sentenceIndex][(tokenIndex+currentIndex+len(currentToken))::]
                            currentIndex = originalText[sentenceIndex].index(currentToken)
                            tokenIndex = currentIndex + len(currentToken)

                    if eachTag == 2:
                        if tagIndex == 0 or eachSentence[tagIndex-1] != 2:
                            currentIndex = originalText[sentenceIndex][tokenIndex::].index(currentToken)
                            originalText[sentenceIndex] = originalText[sentenceIndex][0:tokenIndex] + originalText[sentenceIndex][tokenIndex:(tokenIndex+currentIndex)] + "<ENAMEX TYPE=\"LOCATION\">" + originalText[sentenceIndex][(tokenIndex+currentIndex)::]
                            
                            if (tagIndex != (len(eachSentence)-1) and eachSentence[tagIndex + 1] != 2) or tagIndex == (len(eachSentence)-1):
                                currentIndex = originalText[sentenceIndex][tokenIndex::].index(currentToken)
                                originalText[sentenceIndex] = originalText[sentenceIndex][0:tokenIndex] + originalText[sentenceIndex][tokenIndex:(tokenIndex+currentIndex+len(currentToken))] + "</ENAMEX>" + originalText[sentenceIndex][(tokenIndex+currentIndex+len(currentToken))::]
                            currentIndex = originalText[sentenceIndex].index(currentToken)
                            tokenIndex = currentIndex + len(currentToken)
                        elif tagIndex != (len(eachSentence)-1) and eachSentence[tagIndex - 1] == 2 and eachSentence[tagIndex + 1] == 2:
                            pass
                        elif (tagIndex != (len(eachSentence)-1) and eachSentence[tagIndex + 1] != 2):
                            currentIndex = originalText[sentenceIndex][tokenIndex::].index(currentToken)
                            originalText[sentenceIndex] = originalText[sentenceIndex][0:tokenIndex] + originalText[sentenceIndex][tokenIndex:(tokenIndex+currentIndex+len(currentToken))] + "</ENAMEX>" + originalText[sentenceIndex][(tokenIndex+currentIndex+len(currentToken))::]
                            currentIndex = originalText[sentenceIndex].index(currentToken)
                            tokenIndex = currentIndex + len(currentToken)
                        elif tagIndex == (len(eachSentence)-1):
                            currentIndex = originalText[sentenceIndex][tokenIndex::].index(currentToken)
                            originalText[sentenceIndex] = originalText[sentenceIndex][0:tokenIndex] + originalText[sentenceIndex][tokenIndex:(tokenIndex+currentIndex+len(currentToken))] + "</ENAMEX>" + originalText[sentenceIndex][(tokenIndex+currentIndex+len(currentToken))::]
                            currentIndex = originalText[sentenceIndex].index(currentToken)
                            tokenIndex = currentIndex + len(currentToken)

                    if eachTag == 3:
                        if tagIndex == 0 or eachSentence[tagIndex-1] != 3:
                            currentIndex = originalText[sentenceIndex][tokenIndex::].index(currentToken)
                            originalText[sentenceIndex] = originalText[sentenceIndex][0:tokenIndex] + originalText[sentenceIndex][tokenIndex:(tokenIndex+currentIndex)] + "<ENAMEX TYPE=\"ORGANIZATION\">" + originalText[sentenceIndex][(tokenIndex+currentIndex)::]
                            
                            if (tagIndex != (len(eachSentence)-1) and eachSentence[tagIndex + 1] != 3) or tagIndex == (len(eachSentence)-1):
                                currentIndex = originalText[sentenceIndex][tokenIndex::].index(currentToken)
                                originalText[sentenceIndex] = originalText[sentenceIndex][0:tokenIndex] + originalText[sentenceIndex][tokenIndex:(tokenIndex+currentIndex+len(currentToken))] + "</ENAMEX>" + originalText[sentenceIndex][(tokenIndex+currentIndex+len(currentToken))::]
                            currentIndex = originalText[sentenceIndex].index(currentToken)
                            tokenIndex = currentIndex + len(currentToken)
                        elif tagIndex != (len(eachSentence)-1) and eachSentence[tagIndex - 1] == 3 and eachSentence[tagIndex + 1] == 3:
                            pass
                        elif (tagIndex != (len(eachSentence)-1) and eachSentence[tagIndex + 1] != 1):
                            currentIndex = originalText[sentenceIndex][tokenIndex::].index(currentToken)
                            originalText[sentenceIndex] = originalText[sentenceIndex][0:tokenIndex] + originalText[sentenceIndex][tokenIndex:(tokenIndex+currentIndex+len(currentToken))] + "</ENAMEX>" + originalText[sentenceIndex][(tokenIndex+currentIndex+len(currentToken))::]
                            currentIndex = originalText[sentenceIndex].index(currentToken)
                            tokenIndex = currentIndex + len(currentToken)
                        elif tagIndex == (len(eachSentence)-1):
                            currentIndex = originalText[sentenceIndex][tokenIndex::].index(currentToken)
                            originalText[sentenceIndex] = originalText[sentenceIndex][0:tokenIndex] + originalText[sentenceIndex][tokenIndex:(tokenIndex+currentIndex+len(currentToken))] + "</ENAMEX>" + originalText[sentenceIndex][(tokenIndex+currentIndex+len(currentToken))::]
                            currentIndex = originalText[sentenceIndex].index(currentToken)
                            tokenIndex = currentIndex + len(currentToken)

        return originalText

    def getPredictionElements(self, outputText):
        predicted_corpus = outputText
        yPred_Person, yPred_Location, yPred_Organization = self.text.find_enamex(corpus=predicted_corpus)
        return yPred_Person, yPred_Location, yPred_Organization

    def getTaggedElements(self, Index):
        test_corpus = self.text.corpus[Index[0]:(Index[-1]+1)]
        yPerson, yLocation, yOrganization = self.text.find_enamex(corpus=test_corpus)
        return yPerson, yLocation, yOrganization

    def F1_ExactMatch(self, yPerson, yLocation, yOrganization, yPred_Person, yPred_Location, yPred_Organization):
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

        return F1_ExactMatch_Macro, F1_ExactMatch_Micro
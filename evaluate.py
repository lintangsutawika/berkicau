import re
import csv
import sys
import nltk
import gensim
import string
import numpy as np

from preprocess import findEntity


class evaluate(object):
    """docstring for evaluate"""
    def __init__(self, dir="./Datasets/", filename=None):
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
        outputText = originalText
        for sentenceIndex,eachSentence in enumerate(allPredictions):
            currentIndex = 0
            endTagIndex = (len(eachSentence)-1)
            currentSentence = outputText[sentenceIndex]
            for tagIndex, eachTag in enumerate(eachSentence):
                currentToken = originalTokens[sentenceIndex][tagIndex]
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
                        if tagIndex == 0: #At beginning
                            currentIndex = currentIndex + currentSentence[currentIndex::].index(currentToken) #Get left most index of an entry
                            currentSentence = currentSentence[0:currentIndex] + "<ENAMEX TYPE=\"PERSON\">" + currentSentence[currentIndex::]
                            tempIndex = currentIndex+22+len(currentToken)
                            if eachSentence[tagIndex+1] != 1:
                                currentSentence = currentSentence[0:tempIndex] + "</ENAMEX>" + currentSentence[tempIndex::]
                                currentIndex = tempIndex + 9
                            else:
                                currentIndex = tempIndex
                        elif 0<tagIndex and tagIndex<endTagIndex: #At middle
                            if eachSentence[tagIndex-1] == 1: #Previous word tag is the same
                                if eachSentence[tagIndex+1] == 1: #Next word tag is the same
                                    pass #No need to insert tag
                                elif eachSentence[tagIndex+1] != 1: #Next word tag is NOT the same, the current word is an end word
                                    currentIndex = currentIndex + currentSentence[currentIndex::].index(currentToken) #Get left most index of an entry
                                    currentIndex = currentIndex+len(currentToken)
                                    currentSentence = currentSentence[0:currentIndex] + "</ENAMEX>" + currentSentence[currentIndex::]
                            elif eachSentence[tagIndex-1] != 1: #Previous word tag is NOT the same, the current word is a start word
                                currentIndex = currentIndex + currentSentence[currentIndex::].index(currentToken) #Get left most index of an entry
                                currentSentence = currentSentence[0:currentIndex] + "<ENAMEX TYPE=\"PERSON\">" + currentSentence[currentIndex::]
                                tempIndex = currentIndex+22+len(currentToken)
                                if eachSentence[tagIndex+1] != 1:
                                    currentSentence = currentSentence[0:tempIndex] + "</ENAMEX>" + currentSentence[tempIndex::]
                                    currentIndex = tempIndex + 9
                                else:
                                    currentIndex = tempIndex
                        elif tagIndex == endTagIndex: #At end
                            if eachSentence[tagIndex-1] != 1:
                                currentIndex = currentIndex + currentSentence[currentIndex::].index(currentToken) #Get left most index of an entry
                                currentSentence = currentSentence[0:currentIndex] + "<ENAMEX TYPE=\"PERSON\">" + currentSentence[currentIndex::]
                                tempIndex = currentIndex+22+len(currentToken)
                                currentSentence = currentSentence[0:tempIndex] + "</ENAMEX>" + currentSentence[tempIndex::]
                            elif eachSentence[tagIndex-1] == 1: #Previous word tag is the same
                                currentIndex = currentIndex + currentSentence[currentIndex::].index(currentToken) #Get left most index of an entry
                                currentIndex = currentIndex+len(currentToken)
                                currentSentence = currentSentence[0:currentIndex] + "</ENAMEX>" + currentSentence[currentIndex::]
                    elif eachTag == 2:
                        if tagIndex == 0: #At beginning
                            currentIndex = currentIndex + currentSentence[currentIndex::].index(currentToken) #Get left most index of an entry
                            currentSentence = currentSentence[0:currentIndex] + "<ENAMEX TYPE=\"LOCATION\">" + currentSentence[currentIndex::]
                            tempIndex = currentIndex+24+len(currentToken)
                            if eachSentence[tagIndex+1] != 2:
                                currentSentence = currentSentence[0:tempIndex] + "</ENAMEX>" + currentSentence[tempIndex::]
                                currentIndex = tempIndex + 9
                            else:
                                currentIndex = tempIndex
                        elif 0<tagIndex and tagIndex<endTagIndex: #At middle
                            if eachSentence[tagIndex-1] == 2: #Previous word tag is the same
                                if eachSentence[tagIndex+1] == 2: #Next word tag is the same
                                    pass #No need to insert tag
                                elif eachSentence[tagIndex+1] != 2: #Next word tag is NOT the same, the current word is an end word
                                    currentIndex = currentIndex + currentSentence[currentIndex::].index(currentToken) #Get left most index of an entry
                                    currentIndex = currentIndex+len(currentToken)
                                    currentSentence = currentSentence[0:currentIndex] + "</ENAMEX>" + currentSentence[currentIndex::]

                            elif eachSentence[tagIndex-1] != 2: #Previous word tag is NOT the same, the current word is a start word
                                currentIndex = currentIndex + currentSentence[currentIndex::].index(currentToken) #Get left most index of an entry
                                currentSentence = currentSentence[0:currentIndex] + "<ENAMEX TYPE=\"LOCATION\">" + currentSentence[currentIndex::]
                                tempIndex = currentIndex+24+len(currentToken)
                                if eachSentence[tagIndex+1] != 2:
                                    currentSentence = currentSentence[0:tempIndex] + "</ENAMEX>" + currentSentence[tempIndex::]
                                    currentIndex = tempIndex + 9
                                else:
                                    currentIndex = tempIndex
                        elif tagIndex == endTagIndex: #At end
                            if eachSentence[tagIndex-1] != 2:
                                currentIndex = currentIndex + currentSentence[currentIndex::].index(currentToken) #Get left most index of an entry
                                currentSentence = currentSentence[0:currentIndex] + "<ENAMEX TYPE=\"LOCATION\">" + currentSentence[currentIndex::]
                                tempIndex = currentIndex+24+len(currentToken)
                                currentSentence = currentSentence[0:tempIndex] + "</ENAMEX>" + currentSentence[tempIndex::]
                            elif eachSentence[tagIndex-1] == 2: #Previous word tag is the same
                                currentIndex = currentIndex + currentSentence[currentIndex::].index(currentToken) #Get left most index of an entry
                                currentIndex = currentIndex+len(currentToken)
                                currentSentence = currentSentence[0:currentIndex] + "</ENAMEX>" + currentSentence[currentIndex::]      

                    elif eachTag == 3:
                        if tagIndex == 0: #At beginning
                            currentIndex = currentIndex + currentSentence[currentIndex::].index(currentToken) #Get left most index of an entry
                            currentSentence = currentSentence[0:currentIndex] + "<ENAMEX TYPE=\"ORGANIZATION\">" + currentSentence[currentIndex::]
                            tempIndex = currentIndex+28+len(currentToken)
                            if eachSentence[tagIndex+1] != 3:
                                currentSentence = currentSentence[0:tempIndex] + "</ENAMEX>" + currentSentence[tempIndex::]
                                currentIndex = tempIndex + 9
                            else:
                                currentIndex = tempIndex
                        elif 0<tagIndex and tagIndex<endTagIndex: #At middle
                            if eachSentence[tagIndex-1] == 3: #Previous word tag is the same
                                if eachSentence[tagIndex+1] == 3: #Next word tag is the same
                                    pass #No need to insert tag
                                elif eachSentence[tagIndex+1] != 3: #Next word tag is NOT the same, the current word is an end word
                                    currentIndex = currentIndex + currentSentence[currentIndex::].index(currentToken) #Get left most index of an entry
                                    currentIndex = currentIndex+len(currentToken)
                                    currentSentence = currentSentence[0:currentIndex] + "</ENAMEX>" + currentSentence[currentIndex::]

                            elif eachSentence[tagIndex-1] != 3: #Previous word tag is NOT the same, the current word is a start word
                                currentIndex = currentIndex + currentSentence[currentIndex::].index(currentToken) #Get left most index of an entry
                                currentSentence = currentSentence[0:currentIndex] + "<ENAMEX TYPE=\"ORGANIZATION\">" + currentSentence[currentIndex::]
                                tempIndex = currentIndex+28+len(currentToken)
                                if eachSentence[tagIndex+1] != 3:
                                    currentSentence = currentSentence[0:tempIndex] + "</ENAMEX>" + currentSentence[tempIndex::]
                                    currentIndex = tempIndex + 9
                                else:
                                    currentIndex = tempIndex
                        elif tagIndex == endTagIndex: #At end
                            if eachSentence[tagIndex-1] != 3:
                                currentIndex = currentIndex + currentSentence[currentIndex::].index(currentToken) #Get left most index of an entry
                                currentSentence = currentSentence[0:currentIndex] + "<ENAMEX TYPE=\"ORGANIZATION\">" + currentSentence[currentIndex::]
                                tempIndex = currentIndex+28+len(currentToken)
                                currentSentence = currentSentence[0:tempIndex] + "</ENAMEX>" + currentSentence[tempIndex::]
                            elif eachSentence[tagIndex-1] == 3: #Previous word tag is the same
                                currentIndex = currentIndex + currentSentence[currentIndex::].index(currentToken) #Get left most index of an entry
                                currentIndex = currentIndex+len(currentToken)
                                currentSentence = currentSentence[0:currentIndex] + "</ENAMEX>" + currentSentence[currentIndex::]     
            
            outputText[sentenceIndex] = currentSentence
        return outputText

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
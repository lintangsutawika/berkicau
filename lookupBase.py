import sys
import re
import nltk 
import string
import numpy as np

from evaluate import evaluate
from preprocess import findEntity
from prepareData import prepareData 
from entityTagger import entityTagger

class lookUp(object):
    """docstring for lookUp"""
    def __init__(self, testfile="test.txt", filenameInstansi="ministries.txt", filenameBUMN="bumn.txt", filenameNegara="negara.txt", filenameWilayah="provinsi.txt", dir="./Datasets/", withExtra=False, extrafile="extraTraining.txt"):
        super(lookUp, self).__init__()
        #Gazetter
        self.daftarNegara = re.split('\n',open(dir+filenameNegara, "rb").read())
        self.daftarWilayah = re.split('\n',open(dir+filenameWilayah, "rb").read())
        self.daftarKementrian = re.split('\n',open(dir+filenameInstansi, "rb").read())
        self.daftarBUMN = re.split('\n',open(dir+filenameBUMN, "rb").read())

        self.testCorpus = findEntity(filename=testfile)
        self.tokenizedTest = self.testCorpus.corpus2BIO()[0]
        self.corpus = findEntity().corpus
        self.withExtra = withExtra
        if self.withExtra==True:
            self.extraData = findEntity(filename=extrafile)
            self.extraCorpus = self.extraData.corpus 

    def printProgressBar (self, iteration, total, prefix = '', suffix = '',decimals = 1, length = 100, fill = '#'):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print'\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix),
        sys.stdout.flush()                                                                                                                                                                                                             
        if iteration == total:
            print()

    def getPersonTable(self):
        self.person = []
        for sent in self.corpus:
            _temp = re.findall("<ENAMEX TYPE=\"PERSON\">(.+?)<", sent)
            for element in _temp:
                self.person.append(element.lower())

        if self.withExtra==True:
            for sent in self.extraCorpus:
                _temp = re.findall("<ENAMEX TYPE=\"PERSON\">(.+?)<", sent)
                for element in _temp:
                    self.person.append(element.lower())

        return self.person

    def getLocationTable(self):
        self.location = []
        for sent in self.corpus:
            _temp = re.findall("<ENAMEX TYPE=\"LOCATION\">(.+?)<", sent)
            for element in _temp:
                self.location.append(element.lower())

        if self.withExtra==True:
            for sent in self.extraCorpus:
                _temp = re.findall("<ENAMEX TYPE=\"LOCATION\">(.+?)<", sent)
                for element in _temp:
                    self.location.append(element.lower())

        return self.location

    def getOrganizationTable(self):
        self.organization = []
        for sent in self.corpus:
            _temp = re.findall("<ENAMEX TYPE=\"ORGANIZATION\">(.+?)<", sent)
            for element in _temp:
                self.organization.append(element.lower())

        if self.withExtra==True:
            for sent in self.extraCorpus:
                _temp = re.findall("<ENAMEX TYPE=\"ORGANIZATION\">(.+?)<", sent)
                for element in _temp:
                    self.organization.append(element.lower())

        return self.organization

    def getLookUpPredictions(self):
        person = self.getPersonTable()
        location = self.getLocationTable()
        organization = self.getOrganizationTable()
        #Predict from lookup table

        allPrediction = []
        punct = [" ", ",", ":", "(", ")", ".", "?", "!"]
        for sentenceIndex, sentence in enumerate(self.tokenizedTest):
            self.printProgressBar (sentenceIndex,len(self.tokenizedTest))
            output = np.zeros([len(sentence)], dtype=np.int32)
            for name in person:
                for str1 in punct:
                    for str2 in punct:
                        if str1+name+str2 in self.testCorpus.corpus[sentenceIndex].lower():
                            token = nltk.wordpunct_tokenize(name)
                            index = self.tokenizedTest[sentenceIndex].index(token[0])
                            output[index:index+len(token)] = 1
                            break

            for wilayah in self.daftarWilayah:
                designated = wilayah.lower()
                for str1 in punct:
                    for str2 in punct:
                        if str1+designated+str2 in self.testCorpus.corpus[sentenceIndex].lower():
                            token = nltk.wordpunct_tokenize(designated)
                            index = self.tokenizedTest[sentenceIndex].index(token[0])
                            output[index:index+len(token)] = 2
                            break

            for negara in self.daftarNegara:
                designated = negara.lower()
                for str1 in punct:
                    for str2 in punct:
                        if str1+designated+str2 in self.testCorpus.corpus[sentenceIndex].lower():
                            token = nltk.wordpunct_tokenize(designated)
                            index = self.tokenizedTest[sentenceIndex].index(token[0])
                            output[index:index+len(token)] = 2
                            break

            for loc in location:
                for str1 in punct:
                    for str2 in punct:
                        if str1+loc+str2 in self.testCorpus.corpus[sentenceIndex].lower():
                            token = nltk.wordpunct_tokenize(loc)
                            index = self.tokenizedTest[sentenceIndex].index(token[0])
                            output[index:index+len(token)] = 2
                            break


            for org in organization:
                for str1 in punct:
                    for str2 in punct:
                        if str1+org+str2 in self.testCorpus.corpus[sentenceIndex].lower():
                            token = nltk.wordpunct_tokenize(org)
                            index = self.tokenizedTest[sentenceIndex].index(token[0])
                            output[index:index+len(token)] = 3
                            break

            for bumn in self.daftarBUMN:
                designated = bumn.lower()
                for str1 in punct:
                    for str2 in punct:
                        if str1+designated+str2 in self.testCorpus.corpus[sentenceIndex].lower():
                            token = nltk.wordpunct_tokenize(designated)
                            index = self.tokenizedTest[sentenceIndex].index(token[0])
                            output[index:index+len(token)] = 3
                            break

            for kemen in self.daftarKementrian:
                designated = kemen.lower()
                for str1 in punct:
                    for str2 in punct:
                        if str1+designated+str2 in self.testCorpus.corpus[sentenceIndex].lower():
                            token = nltk.wordpunct_tokenize(designated)
                            index = self.tokenizedTest[sentenceIndex].index(token[0])
                            output[index:index+len(token)] = 3
                            break

            for wordTag, word in enumerate(sentence):
                if word in person and output[wordTag] != 1:
                    output[wordTag] = 1
                if word in location and output[wordTag] != 2:
                    output[wordTag] = 2
                if word in organization and output[wordTag] != 3:
                    output[wordTag] = 3

            allPrediction.append(output)
        return allPrediction


if __name__ == '__main__':

    evaluation = evaluate(filename="test.txt")
    lookUpPred = lookUp()
    allPrediction = lookUpPred.getLookUpPredictions()
    text = evaluation.getProperOutput(allPrediction, [0,len(tokenizedTest)])
    with open("output_LookUpTable_ExtraCorpus.txt", 'w') as txtFile:
        for lines in text:
            txtFile.write(lines+"\r\n")

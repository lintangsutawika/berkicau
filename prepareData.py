import re
import csv
import sys
import nltk
import gensim
import _pickle as cPickle
import numpy as np

import POStagger
from preprocess import findEntity

class prepareData(object):
    """docstring for prepareData"""
    def __init__(self,mode="None"):
        super(prepareData, self).__init__()
        self.mode = mode
        self.text = findEntity()

        #Load Trained Tagger 
        self.posTagger = POStagger.POStagger()
        self.tnt = self.posTagger.restore_model()

    def getDefaultData(self):
        toFeed = []
        data, tags = self.text.corpus2BIO(mode=self.mode)
        for eachSentence in data:
            toFeed.append(eachSentence)
        return toFeed

    def getDefaultTag(self):
        tagFeed = []
        data, tags = self.text.corpus2BIO(mode=self.mode)
        for eachSentence in data:
            tagged = self.tnt.tag(eachSentence)
            tempTag = []
            for tagging in tagged:
                tempTag.append(tagging[1])
            tagFeed.append(tempTag)

        return tagFeed

    def getTestData(self, corpus):
        toFeed = []
        data, tags = self.text.corpus2BIO(mode=self.mode, corpus=corpus)
        for eachSentence in data:
            toFeed.append(eachSentence)
        return toFeed

    def getTestTag(self, rawSentence):
        tagFeed = []
        for sentences in rawSentence:
            tagged = self.tnt.tag(sentences)
            tempTag = []
            for tagging in tagged:
                tempTag.append(tagging[1])
            
            tagFeed.append(tempTag)

        return tagFeed
    
    def getKalimat1000Data(self,toFeed):
        #Gain large corpus of Indonesian sentences
        for sentence in self.posTagger.preprocessing():
            sent = []
            for eachword in sentence:
                word,tag = eachword
                sent.append(word)
            toFeed.append(sent)
        return toFeed

    def getKalimat1000Tag(self,toTag):
        #Gain large corpus of Indonesian sentences
        for sentence in self.posTagger.preprocessing():
            tags = []
            for eachword in sentence:
                word,tag = eachword
                tags.append(tag)
            toTag.append(tags)
        return toTag

    def getTweetData(self, toFeed, filename="MoreTweets.tsv"):
        #Gain large corpus of tweets
        rawSentence = []
        with open("Datasets/"+filename, 'rU') as csvfile:
            spamreader = csv.reader(csvfile, delimiter='\n', quotechar='|')
            for spam in spamreader:
                rawSentence.append(spam)

        corpusSentence =[]
        for individualSentence in rawSentence:
            if individualSentence == []:
                pass
            else:
                corpusSentence.append(individualSentence[0])

        # corpusSentence = self.text.removeAll(corpusSentence)
        corpusSentence = self.text.corpus2BIO(mode=self.mode, corpus=corpusSentence)[0]

        for sentences in corpusSentence:
            # token = nltk.wordpunct_tokenize(sentences.lower())
            toFeed.append(sentences)

        return toFeed

    def getTweetTag(self, tagFeed, filename="MoreTweets.tsv"):
        rawSentence = []
        with open("Datasets/"+filename, 'rU') as csvfile:
            spamreader = csv.reader(csvfile, delimiter='\n', quotechar='|')
            for spam in spamreader:
                rawSentence.append(spam)

        corpusSentence =[]
        for individualSentence in rawSentence:
            if individualSentence == []:
                pass
            else:
                corpusSentence.append(individualSentence[0])

        corpusSentence = self.text.removeAll(corpusSentence)

        for sentences in corpusSentence:
            token = nltk.wordpunct_tokenize(sentences.lower())
            tagged = self.tnt.tag(token)
            tempTag = []
            for tagging in tagged:
                tempTag.append(tagging[1])
            
            tagFeed.append(tempTag)

        return tagFeed

    def getWord2Vec(self, toFeed, dim=50):
        dimension = dim
        corpusTweet = toFeed
        word2vec = gensim.models.Word2Vec(corpusTweet, min_count=1,  size=dimension)

        return word2vec

    def getTag2Vec(self, tagFeed, dim=50):
        dimension = dim
        corpusTag = tagFeed
        tag2vec = gensim.models.Word2Vec(corpusTag, min_count=1,  size=dimension)

        return tag2vec

    def save_model(self, dataObject, path):
        cPickle.dump( dataObject, open( path, "wb" ))

    def restore_model(self, path):
        object_path = open(path, 'r')
        objectToRestore = cPickle.Unpickler(object_path)
        dataset = objectToRestore.load()
        object_path.close()

        return dataset

if __name__ == '__main__':
    nerData = prepareData()
    #toFeed = nerData.getDefaultData()
    #tagFeed = nerData.getDefaultTag()

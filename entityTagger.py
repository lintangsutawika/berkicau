import re
import nltk
import POStagger
from preprocess import findEntity
import numpy as np
import gensim
import csv 
from sklearn.model_selection import KFold
from Bi_LSTM import BiLSTM_CRF
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

#Load Trained Tagger 
posTagger = POStagger.POStagger()
tnt = posTagger.restore_model()

text = findEntity()
tags, data = text.corpus2BIO()
toFeed = []
for eachSentence in data:
	toFeed.append(eachSentence)

#Gain large corpus
rawSentence = []
with open("Training/MoreTweets.tsv", 'rU') as csvfile:
	spamreader = csv.reader(csvfile, delimiter='\n', quotechar='|')
	for spam in spamreader:
		rawSentence.append(spam)

corpusSentence =[]
for individualSentence in rawSentence:
	if individualSentence == []:
		pass
	else:
		corpusSentence.append(individualSentence[0])

for sentences in corpusSentence:
	token = nltk.wordpunct_tokenize(sentences.lower())
	toFeed.append(token)

corpusTweet = toFeed
word2vec = gensim.models.Word2Vec(corpusTweet, min_count=1,  size=50)
#Stemming
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 50
HIDDEN_DIM = 40

tag_to_ix = {"None": 0, "I-LOC": 1, "I-ORG": 2, "I-PER":3, START_TAG: 4, STOP_TAG: 5}

model = BiLSTM_CRF(100, tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

kf = KFold(n_splits=5)
for train_index, test_index in kf.split(data):
	X_train, X_test = np.asarray(data)[train_index], np.asarray(data)[test_index]
	y_train, y_test = np.asarray(tags)[train_index], np.asarray(tags)[test_index]

	# model.wv['computer']
	# Make sure prepare_sequence from earlier in the LSTM section is loaded
	for epoch in range(10):  # again, normally you would NOT do 300 epochs, it is toy data
	    for index,sentenceTrain in enumerate(X_train):
	        # Step 1. Remember that Pytorch accumulates gradients.
	        # We need to clear them out before each instance
	        model.zero_grad()

	        # Step 2. Get our inputs ready for the network, that is,
	        # turn them into Variables of word indices.
	        embedded_sentence = torch.from_numpy(np.asarray([word2vec.wv[w] for w in sentenceTrain]))
	        targets = torch.LongTensor([t for t in y_train[index]])

	        # Step 3. Run our forward pass.
	        neg_log_likelihood = model.neg_log_likelihood(autograd.Variable(embedded_sentence), targets)
	        print("Epoch: {}, sentence_ind: {}, neg_log: {}".format(epoch, index, neg_log_likelihood))
	        # Step 4. Compute the loss, gradients, and update the parameters by
	        # calling optimizer.step()
	        neg_log_likelihood.backward()
	        optimizer.step()

	    for index,testTrain in enumerate(X_test):
			embedded_sentence = torch.from_numpy(np.asarray([word2vec.wv[w] for w in testTrain]))
			printSentence = ""
			for w in testTrain:
				printSentence = printSentence + " " + w
			targets = torch.LongTensor([t for t in y_test[index]])
			output = model(autograd.Variable(embedded_sentence))[1]
			print("Sentence:\n{}\nTag:\n{}\nPredict:\n{}".format(printSentence, targets.numpy(), np.asarray(output)))

# We got it!
#While there are consecutives NNP, make that inot 1 entity
#if a[i][1] = NNP
#while a[i][1] == "NNP":
#	
#Argmax distance in word to vec
# distance.nlevenshtein(method=1)shortest alignment
#Use for alay words
#People
#	-Public figures
#	-Names
#	-Characters

#1 MORE DATA FOR WORD EMBEDDING
#2 BI-LSTM
#3 TRAINING DATA changed to Entity of No entity
#4 Output dcoument that prints word as an entity of not

#Evaluation
# yPerson, yLocation, yOrganization = text.find_enamex(corpus=test_corpus)
# yPred_Person, yPred_Location, yPred_Organization = text.find_enamex(corpus=predicted_corpus)

true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

for predictions in yPred_Person:
	if (predictions in yPerson) == True:
		true_positive += 1
	else:
		false_positive +=1

for predictions in yPred_Location:
	if (predictions in yLocation) == True:
		true_positive += 1
	else:
		false_positive +=1

for predictions in yPred_Organization:
	if (predictions in yOrganization) == True:
		true_positive += 1
	else:
		false_positive +=1

false_negative = (len(yPerson) + len(yLocation) + len(yOrganization)) - true_positive

#Metrics
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
F1 = 2 * true_positive / (2* true_positive + false_positive + false_negative)

import glob
import logging
import logging.handlers
import sys
import re
import nltk
import POStagger
from preprocess import findEntity
import numpy as np
import gensim
import csv 
from sklearn.model_selection import KFold
import Bi_LSTM
from Bi_LSTM import BiLSTM_CRF
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from datasetNER import datasetNER

def printProgressBar (iteration, total, prefix = '', suffix = '',decimals = 1, length = 100, fill = '#'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print'\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix),
    sys.stdout.flush()
    # Print New Line on Complete                                                                                                                                                                                                              
    if iteration == total:
        print()

LOG_FILENAME = 'NER_logging.out'
# Set up a specific logger with our desired output level
# logging = logging.getLogger('logging')
logging.basicConfig(filemode = 'a', level=logging.DEBUG, filename=LOG_FILENAME)

use_gpu = torch.cuda.is_available()

#Load Trained Tagger 
posTagger = POStagger.POStagger()
tnt = posTagger.restore_model()

text = findEntity()
data, tags = text.corpus2BIO(mode="withIntermediate")
toFeed = []
for eachSentence in data:
    toFeed.append(eachSentence)

#Gain large corpus
rawSentence = []
with open("Datasets/MoreTweets.tsv", 'rU') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\n', quotechar='|')
    for spam in spamreader:
        rawSentence.append(spam)

corpusSentence =[]
for individualSentence in rawSentence:
    if individualSentence == []:
        pass
    else:
        corpusSentence.append(individualSentence[0])

corpusSentence = text.removeAll(corpusSentence)

for sentences in corpusSentence:
    token = nltk.wordpunct_tokenize(sentences.lower())
    toFeed.append(token)

corpusTweet = toFeed
word2vec = gensim.models.Word2Vec(corpusTweet, min_count=1,  size=100)

#Stemming
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 100
HIDDEN_DIM = 50
# tag_to_ix = {"None": 0, "I-LOC": 1, "I-ORG": 3, "I-PER":5, START_TAG: 7, STOP_TAG: 8} #Version 1
tag_to_ix = {"None": 0, "B-LOC":1, "I-LOC":2, "B-ORG":3, "I-ORG":4, "B-PER":5, "I-PER": 6, START_TAG:7, STOP_TAG:8} #Version 2

model = BiLSTM_CRF(100, tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-4, momentum=0.9, nesterov=True)

kf = KFold(n_splits=5)
for nFold, (train_index, test_index) in enumerate(kf.split(data)):
    X_train, X_test = np.asarray(data)[train_index], np.asarray(data)[test_index]
    y_train, y_test = np.asarray(tags)[train_index], np.asarray(tags)[test_index]

    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(500):  # again, normally you would NOT do 300 epochs, it is toy data
        
        #Make in to batch of 10?
        # dsets = {'train':tnt.dataset.ListDataset(datasetNER()),
        #          'val'  :val}

        # dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=5,shuffle=True, num_workers=4) for x in ['train', 'val']}
        # wordData = 
        batchX = np.array_split(X_train, 6)
        batchY = np.array_split(y_train, 6)
        epochLoss = 0
        for batchInd,batchElement in enumerate(batchX):
            totalLoss = 0
            for index,sentenceTrain in enumerate(batchElement): #dsets_loaders['train']
                optimizer = Bi_LSTM.exp_lr_scheduler(optimizer, epoch)
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                model.zero_grad()

                # Step 2. Get our inputs ready for the network, that is,
                # turn them into Variables of word indices.
                embedded_sentence = torch.from_numpy(np.asarray([word2vec.wv[w] for w in sentenceTrain]))
                targets = torch.LongTensor([t for t in batchY[batchInd][index]])

                # if use_gpu:
                #     inputs, labels = autograd.Variable(embedded_sentence.cuda()), autograd.Variable(targets.cuda())
                # else:
                inputs, labels = autograd.Variable(embedded_sentence), autograd.Variable(targets)
                # Step 3. Run our forward pass.
                neg_log_likelihood = model.neg_log_likelihood(inputs, targets) #/batchSize
                totalLoss += neg_log_likelihood
                Log = "nFold: {}, Epoch: {}, Batch: {}\n".format(nFold, epoch, batchInd)
                print(Log)
                printProgressBar (index, 63)
                # print '\r[{0}]'.format('#'*(index/63))
                logging.debug(Log)
                # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            epochLoss += totalLoss
            gradientLoss = totalLoss/16
            gradientLoss.backward()
            # neg_log_likelihood.backward()
            optimizer.step()

        
        confusionMatrix = np.zeros([7,7])
        valLoss = 0
        for index,testTrain in enumerate(X_test):
            embedded_sentence = torch.from_numpy(np.asarray([word2vec.wv[w] for w in testTrain]))
            printSentence = ""
            for w in testTrain:
                printSentence = printSentence + " " + w

            targets = torch.LongTensor([t for t in y_test[index]])
            inputs, labels = autograd.Variable(embedded_sentence), autograd.Variable(targets)
            valLoss += model.neg_log_likelihood(inputs, targets)
            output = np.asarray(model(autograd.Variable(embedded_sentence))[1])
            for indices, tag in enumerate(targets.numpy()):
                confusionMatrix[tag, output[indices]] += 1 
            # print("Sentence:\n{}\nTag:\n{}\nPredict:\n{}".format(printSentence, targets.numpy(), np.asarray(output)))
            print("Sentence:\n{}\nTag:\n{}\nPredict:\n{}".format(printSentence, targets.numpy(), output))

        trainLoss = epochLoss/np.shape(X_train)[0]
        validationLoss = valLoss/np.shape(X_test)[0]
        print("Epoch: {}, TrainLoss: {}, ValidationLoss: {}".format(epoch, trainLoss.data.numpy(), validationLoss.data.numpy()))
        with open("performance.csv", 'a') as csvFile:
            writer = csv.writer(csvFile)
            # writer.writerow(["Epoch","TrainLoss","ValidationLoss","None","B-LOC","I-LOC","B-ORG","I-ORG","B-PER","I-PER"])
            writer.writerow([nFold, epoch, trainLoss.data.numpy()[0], validationLoss.data.numpy()[0]]) #"None","B-LOC","I-LOC","B-ORG","I-ORG","B-PER","I-PER"])
            csvFile.close()
        
# We got it!
#While there are consecutives NNP, make that inot 1 entity
#if a[i][1] = NNP
#while a[i][1] == "NNP":
#   
#Argmax distance in word to vec
# distance.nlevenshtein(method=1)shortest alignment
#Use for alay words
#People
#   -Public figures
#   -Names
#   -Characters

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

import re
import nltk
import POStagger
import preprocess
import numpy as np
import tensorflow as tf

#Load Trained Tagger 
posTagger = POStagger.POStagger()
tnt = posTagger.restore_model()

#Tokenization
text = preprocess.findEntity()
filtered = text.removeAll()
toFeed = []
for sentences in filtered:
	token = nltk.wordpunct_tokenize(sentences.lower())
	toFeed.append(token)
	
tags = tnt.tag(toFeed[0])
for tag in tags:
	if tag

#Stemming

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

#Model Goes through 2 stages of NER
#1 Machine Learning
#2 Rule Based filtering
#3 Final tagging (Determine whether a tag is loc, org, or person) Rule based as well


#Observations
#-Tanda titik dua sering kali menjadi penanda quote sehingga frase yang mendahului titik dua sering kali adalah orang, i.e Jokowi dst
#-Jalan bisa jadi "Jl." atau "Jln." atau "jln." dan "jl." atau "Jalan" "jl" "Jl"
Jalanan = ["Jl", "jl", "JL", "Jln", "jln", "JLN"]
nama_jalan = []
for kata_jalan in Jalanan:
	if any(kata_jalan in word for word in sentence) == True:
		break
#
#News data => Use trigram
#Alay data => Word2Vec
#Use u.screen_name to get screen_name
# for page in tweepy.Cursor(api.followers_ids, screen_name="lintangsutawika").pages():
	# ids.extend(page)
	# time.sleep(60)
#Coocurance

#KBBI curl function
# https://kbbi.kemdikbud.go.id/entri
#Di, nya - cut function
#di + unk, mostlikely a place
#Pada berita, kata kerja yang dipakai tidak berimbuhan

#Evaluation
yPerson, yLocation, yOrganization = text.find_enamex(corpus=test_corpus)
yPred_Person, yPred_Location, yPred_Organization = text.find_enamex(corpus=predicted_corpus)

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

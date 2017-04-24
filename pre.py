import re
import preprocess
import nltk

#ToDo
#Tokenization
text = preprocess.findEntity()
filtered = text.removeAll()
toFeed = []
for sentences in filtered:
	token = nltk.wordpunct_tokenize(sentences)
	print(token)
	toFeed.append(token)
#Argmax distance in word to vec
# distance.nlevenshtein(method=1)shortest alignment
#Use for alay words
#People
#	-Public figures
#	-Names
#	-Characters
                    
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

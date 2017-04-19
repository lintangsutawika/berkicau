import re
import find_enamex
import nltk

#ToDo
#Tokenization
text = find_enamex.findEntity()
filtered = text.removeAll()
for sentences in filtered:
	token = nltk.wordpunct_tokenize(sentences)
	print(token)

#Observations
#-Tanda titik dua sering kali menjadi penanda quote sehingga frase yang mendahului titik dua sering kali adalah orang, i.e Jokowi dst
#Coocurance
import nltk
import re
from nltk import word_tokenize
from nltk.tag import tnt

class POStagger(object):
	def _init_(self,dataset='./Training/Dataset_10000_kalimat.tsv'):
		
	def preprocessing(path):
		f = open(path)
		raw = f.read()
		Segmented_pre_sentence = re.split('</kalimat>',raw)
		Segmented_pre_sentence.pop(-1)
		sentence_data = []
		tagged_sentence = []
		for i,each_sentence in enumerate(Segmented_pre_sentence):
			each_word = re.split('\n',each_sentence)
			while True:
				try:
					each_word.pop(each_word.index(''))
					each_word.pop(0)
				except ValueError:
					break

			for tags in each_word:
				tag_elements = re.split('\t',tags)
				tag_tuple = (tag_elements[0],tag_elements[1])
				tagged_sentence.append(tag_tuple)
			tagged_sentence = []

			sentence_data.append(tagged_sentence)

		return sentence_data

	corpus_data = preprocessing('./Training/Dataset_10000_kalimat.tsv')
	train_data = corpus_data[:9000]
	test_data1 = corpus_data[9000:]

	tnt_pos_tagger = tnt.TnT()
	tnt_pos_tagger.train(train_data)
	score1 = tnt_pos_tagger.evaluate(test_data1)
	print("Score1: {}\n".format(score1))
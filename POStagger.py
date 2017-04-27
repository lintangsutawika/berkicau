import nltk
import re
from nltk import word_tokenize
from nltk.tag import tnt
import cPickle

class POStagger(object):
	def __init__(self,dataset='./Training/Dataset_10000_kalimat.tsv'):
		self.dataset = dataset

	def preprocessing(self, dataset=None):
		if dataset == None:
			dataset = self.dataset

		f = open(dataset)
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

	def save_model(self, object=tnt_pos_tagger, path="tnt_pos_tagger.pic"):
		cPickle.dump( object, open( path, "wb" ))

	def restore_model(self, path="tnt_pos_tagger.pic"):
		object_path = open("tnt_pos_tagger.pic", 'r')
		tag_object = cPickle.Unpickler(object_path)
		tagger = tag_object.load()
		tagger_path.close()

		return tagger
	
if __name__ == '__main__':
	posTagger = POStagger()
	corpus_data = posTagger.preprocessing()
	train_data = corpus_data[:9000]
	test_data = corpus_data[9000:]

	tnt_pos_tagger = tnt.TnT()
	tnt_pos_tagger.train(train_data)
	score = tnt_pos_tagger.evaluate(test_data1)
	print("Trained Model Score: {}\n".format(score))

	#Save Model
	posTagger.save_model()

	loaded_tagger = posTagger.restore_model()
	
	score = loaded_tagger.evaluate(test_data)
	print("Restored Tagger Score: {}\n".format(score))

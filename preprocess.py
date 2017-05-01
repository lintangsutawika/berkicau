import re
import string
import nltk

class findEntity(object):
	"""docstring for findEntity"""
	def __init__(self, dir="./Training/",filename="training_data_new.txt"):
		self.corpus = re.split('\r\n',open(dir+filename, "rb").read())
		self.corpus.pop(-1)
		self.Types = ["PERSON","LOCATION","ORGANIZATION"]
		self.person = []
		self.location = []
		self.organization = []
		self.typeDicts = {
			"PERSON":self.person,
			"LOCATION":self.location,
			"ORGANIZATION":self.organization
			}	

	def get_corpus(self):
		return self.corpus

	def find_enamex(self, corpus=None):
		if corpus== None:
			corpus = self.corpus

		for type in self.Types:
			start_string = "<ENAMEX TYPE=\"{}\">".format(type)
			for _temp in corpus:
				while True:
					try:
						start = _temp.index(start_string) + len(start_string)
						end = start+_temp[start:].index("</ENAMEX>")
						self.typeDicts["{}".format(type)].append(_temp[start:end])
						_temp = _temp[end+len("</ENAMEX>"):]
					except ValueError:
						break
		return [self.person, self.location, self.organization]

	def renameUser(self,corpus=None):
		_new = []
		if corpus == None:
			corpus = self.corpus
		
		for _temp in corpus:
			_temp =  re.sub( r'(^|[^@\w])@(\w{1,15})\b','',_temp)
			_new.append(_temp)

		return _new
	
	def removeEnamex(self,corpus=None):
		_new = []
		if corpus == None:
			corpus = self.corpus
			
		for _temp in corpus:
			_temp =  re.sub( r'</ENAMEX>',' ',_temp)
			for type in self.Types:
				_temp =  re.sub( r'<ENAMEX TYPE="{}">'.format(type),' ',_temp)
			_new.append(_temp)

		return _new

	def addSpacingEnamex(self,corpus=None):
		_new = []
		if corpus == None:
			corpus = self.corpus
			
		for _temp in corpus:
			_temp =  re.sub( r'</ENAMEX>','</ENAMEX> ',_temp)
			# for type in self.Types:
			# 	_temp =  re.sub( r'<ENAMEX TYPE="{}">'.format(type),'<ENAMEX TYPE="{}"> '.format(type),_temp)
			_new.append(_temp)

		return _new
	
	def removeHashtag(self, corpus=None):
		_new = []
		if corpus == None:
			corpus = self.corpus

		for _temp in corpus:
			_temp = re.sub(r'#(\w+)', '', _temp)
			_new.append(_temp)

		return _new

	def removeURL(self, corpus=None):
		#https://support.twitter.com/articles/78124
		_new = []
		if corpus == None:
			corpus = self.corpus
	
		for _temp in corpus:
			_temp = re.sub(r'http:\S+', '', _temp, flags=re.MULTILINE)
			_temp = re.sub(r'https:\S+', '', _temp, flags=re.MULTILINE)
			_new.append(_temp)

		return _new

	def removeEmoticon(self, corpus=None):
		#https://support.twitter.com/articles/78124
		_new = []
		emoticons_str = r"(?:[:=;B][oO\-]?[D\)\]\(\]/\\OpP])"

		if corpus == None:
			corpus = self.corpus
	
		for _temp in corpus:
			_temp = re.sub(emoticons_str, '', _temp)
			_new.append(_temp)

		return _new

	def removeSingleLetter(self, corpus=None):
		pass

	def removeRepeatLetter(self, corpus=None):
		_new = []
		if corpus == None:
			corpus = self.corpus

		for _temp in corpus:
			repeats = re.findall(r'((\w)\2{1,})', _temp)
			for repeatGroup in repeats:
				_temp = re.sub(repeatGroup[0], repeatGroup[1], _temp)
			
			_new.append(_temp)

		return _new

	def removeAll(self,corpus=None):
		if corpus == None:
			corpus = self.corpus
		
		_temp = self.removeURL()
		_temp = self.renameUser(_temp)
		_temp = self.removeEnamex(_temp)
		_temp = self.removeHashtag(_temp)
		_temp = self.removeEmoticon(_temp)
		return _temp

	def corpus2BIO(self, corpus=None):
		if corpus== None:
			corpus = self.corpus

		_temp = self.removeURL()
		_temp = self.renameUser(_temp)
		_temp = self.removeHashtag(_temp)
		_temp = self.removeEmoticon(_temp)
		_temp = self.addSpacingEnamex(_temp)

		tags = []
		dataset = []
		for sentences in _temp:
			sentences = sentences.lower()
			tempSentence = re.split(' +', sentences)

			while '<enamex' in tempSentence:
				tempSentence.pop(tempSentence.index('<enamex'))

			while '' in tempSentence:
				tempSentence.pop(tempSentence.index(''))

			punctuations = list(string.punctuation)
			tempSentence = [word for word in tempSentence if word not in punctuations]
			
			tagSentence = []
			for index, words in enumerate(tempSentence):
				if "type=\"person\"" in words:
					# if "</ENAMEX>" in words:
					tempSentence[index] =  re.sub( "type=\"person\">",'',words)
					tagSentence.append(1)
				elif "type=\"location\"" in words:
					# if "</ENAMEX>" in words:
					tempSentence[index] =  re.sub( "type=\"location\">",'',words)
					tagSentence.append(2)
				elif "type=\"organization\"" in words:
					# if "</ENAMEX>" in words:
					tempSentence[index] =  re.sub( "type=\"organization\">",'',words)
					tagSentence.append(3)
				elif "</enamex>" in words:
					tempSentence[index] =  re.sub( "</enamex>",'',words)
					if index > 0:
						tagSentence.append(tagSentence[index-1])
				else:
					if index > 0 and tagSentence[index-1] != 0 and "</enamex>" not in tempSentence[index-1]: 
						tagSentence.append(tagSentence[index-1])
					else:
						tempSentence[index-1] = re.sub( "</enamex>",'',tempSentence[index-1])
						tagSentence.append(0)

			tags.append(tagSentence)
			dataset.append(tempSentence)
		return [tags, dataset]
					# 	else:
					# if index == 0:
					# 	tagSentence.append(0)
					# elif tagSentence[index-1] != 0 and ("</ENAMEX>" in tagSentence[index-1]):
					# 	tagSentence.append(tagSentence[index-1])
					# else:
					# 	tagSentence.append(0)						
		#TagSET
		#None :0
		#I-LOC:1
		#I-ORG:2
		#I-PER:3


if __name__ == '__main__':
	text = findEntity()
	filtered = text.removeAll()
	file = open("output.txt", "w")
	for sentences in filtered:
		file.write(sentences) 
	file.close()
import re

class findEntity(object):
	"""docstring for findEntity"""
	def __init__(self, dir="./Training/",filename="training_data_new.txt"):
		self.corpus = re.split('\r\n',open(dir+filename, "rb").read())
		self.corpus.pop(-1)

	def get_corpus(self):
		return self.corpus

	def find_enamex(self, corpus=None):
		self.Types = ["PERSON","LOCATION","ORGANIZATION"]
		self.person = []
		self.location = []
		self.organization = []
		self.typeDicts = {
			"PERSON":self.person,
			"LOCATION":self.location,
			"ORGANIZATION":self.organization
			}		
		
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
			_temp =  re.sub( r'</ENAMEX>','',_temp)
			for type in Types:
				_temp =  re.sub( r'<ENAMEX TYPE="{}">'.format(type),'',_temp)
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
		
		_temp = self.renameUser()
		_temp = self.removeEnamex(_temp)
		_temp = self.removeHashtag(_temp)
		_temp = self.removeURL(_temp)
		_temp = self.removeEmoticon(_temp)
		return _temp


if __name__ == '__main__':
	text = findEntity()
	filtered = text.removeAll()
	file = open("output.txt", "w")
	for sentences in filtered:
		file.write(sentences) 
	file.close()
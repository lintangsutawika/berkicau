import re

Types = ["PERSON","LOCATION","ORGANIZATION"]
person = []
location = []
organization = []
typeDicts = {
	"PERSON":person,
	"LOCATION":location,
	"ORGANIZATION":organization
	}

class findEntity(object):
	"""docstring for findEntity"""
	def __init__(self, dir="./Training/",filename="training_data_new.txt"):
		self.corpus = re.split('\r\n',open(dir+filename, "rb").read())
		self.corpus.pop(-1)
		
	def get_corpus(self):
		return self.corpus

	def find_enamex(self, corpus=None):
		if corpus== None:
			corpus = self.corpus

		for type in Types:
			start_string = "<ENAMEX TYPE=\"{}\">".format(type)
			for _temp in corpus:
				while True:
				    try:
	   					start = _temp.index(start_string) + len(start_string)
	   					end = start+_temp[start:].index("</ENAMEX>")
						typeDicts["{}".format(type)].append(_temp[start:end])
						_temp = _temp[end+len("</ENAMEX>"):]
				    except ValueError:
				    	break

if __name__ == '__main__':
	text = findEntity()
	tweets = text.get_corpus()
	text.find_enamex()
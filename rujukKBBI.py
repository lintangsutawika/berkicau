import pycurl
from StringIO import StringIO
from bs4 import BeautifulSoup

class rujukKBBI(object):
	def __init__(self):
		self.bufferString = StringIO() #Initialize buffer for string of html
		self.crawler = pycurl.Curl() #Initialize curl object
		
	def cari_kata(self, kata='berkicau'):
		self.crawler.setopt(self.crawler.URL, 'https://kbbi.kemdikbud.go.id/entri/' + kata)
		self.crawler.setopt(self.crawler.WRITEDATA, self.bufferString)
		self.crawler.perform()
		self.crawler.close()

		html_doc = self.bufferString.getvalue()
		soup = BeautifulSoup(html_doc, 'html.parser')
		self.span_tags = soup.find_all('span')
		for tags in self.span_tags:
			if tags.has_attr('class'):
				if any("text-danger" in tag_element for tag_element in tags['class']): #Easiest sign to know that the word does not exist
					return None
		return self.span_tags

	def kata_dasar(self, soup_object):
		pass

	def jenis_kata(self, kata='berkicau'):
		try:
			for tags in self.cari_kata(kata):
				if tags.has_attr('title'):
					if tags['title'] == "Verba: kata kerja":
						return "VB"
					elif tags['title'] == "Nomina: kata benda":
						return "NN"
					elif tags['title'] == "Adjektiva: kata yang menjelaskan nomina atau pronomina":
						return "JJ"
		except TypeError:
			return "NOTFOUND"


if __name__ == '__main__':
	rujukan = rujukKBBI()
	# soup = rujukan.cari_kata()
	a = rujukan.jenis_kata()
	print(a)
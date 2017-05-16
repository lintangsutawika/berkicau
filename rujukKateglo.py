import urllib
import contextlib
from bs4 import BeautifulSoup

class rujukKateglo(object):
	def __init__(self):
		pass

	def jenis_kata(self, kata='berkicau'):
		# self.crawler = pycurl.Curl() #Initialize curl object
		urlToOpen = 'http://kateglo.com/index.php?op=3&phrase='+kata+'&lex=&type=&src=&mod=dictionary&srch=Cari'
		with contextlib.closing(urllib.urlopen(urlToOpen)) as html_doc:
			soup = BeautifulSoup(html_doc, 'html.parser')
			self.span_tags = soup.find_all('span')
		
			for tags in self.span_tags:
				if len(tags.contents) > 0:	
					if str(tags.contents[0]) == 'v':
						return "VB"
					if str(tags.contents[0]) == 'n':
						return "NN"
					if str(tags.contents[0]) == 'adj':
						return "JJ"
					if str(tags.contents[0]) == 'kp':
						return "KP"
					if str(tags.contents[0]) == 'pre':
						return "P"
			return "NOTFOUND"
		
	def kata_dasar(self, soup_object):
		pass

if __name__ == '__main__':
	kata = "ajak"
	rujukan = rujukKateglo()
	jenis = rujukan.jenis_kata(kata)
	print("Jenis kata untuk {} adalah {}".format(kata,jenis))
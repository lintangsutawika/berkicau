import re
import string
import nltk

class findEntity(object):
    """docstring for findEntity"""
    def __init__(self, dir="./Datasets/",filename="training_data_new.txt"):
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
            _temp =  re.sub( r'</ENAMEX>',' _ENAMEX ',_temp)
            for type in self.Types:
                _temp =  re.sub( r'<ENAMEX TYPE=\"{}\">'.format(type),'TYPE_{}_'.format(type),_temp)
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
        emoticons_str = r"(?:[:=;B\-][oO\"\_\-]?[\-D\)\]\(\]/\\Op3]{2,3})"

        if corpus == None:
            corpus = self.corpus
    
        for _temp in corpus:
            _temp = re.sub(emoticons_str, '', _temp)
            _temp = re.sub(r'[^\x00-\x7F]', '', _temp)
            _new.append(_temp)

        return _new

    def removeSingleLetter(self, corpus=None):
        pass

    def addSpaceToPunctuation(self, corpus=None):
        _new = []
        punctuations = list(string.punctuation)
        if corpus == None:
            corpus = self.corpus

        for _temp in corpus:
            for pun in punctuations:
                _temp = re.sub(pun, " "+pun+" ", _temp)
            _new.append(_temp)

        return _new

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
        
        _temp = self.removeURL(corpus)
        _temp = self.renameUser(_temp)
        # _temp = self.removeEnamex(_temp)
        _temp = self.removeHashtag(_temp)
        _temp = self.removeEmoticon(_temp)
        _temp = self.addSpacingEnamex(_temp)
        return _temp

    def corpus2BIO(self, mode="withIntermediate" ,corpus=None):
        if corpus == None:
            corpus = self.corpus
            _temp = self.removeURL()

        # elif type(corpus) == tuple:
            # _tempData, _tempTags = corpus
            # _temp = self.removeURL(_tempData)

        else:
            _tempData = corpus
            _temp = self.removeURL(_tempData)

        _temp = self.renameUser(_temp)
        _temp = self.removeHashtag(_temp)
        _temp = self.removeEmoticon(_temp)
        _temp = self.addSpacingEnamex(_temp)

        #Tagging index
        #"None"     : 0, 
        #"B-LOC"    : 1, 
        #"I-LOC"    : 2,
        #"B-ORG"    : 3, 
        #"I-ORG"    : 4,
        #"B-PER"    : 5,          
        #"I-PER"    : 6, 
        #START_TAG  : 7, 
        #STOP_TAG   : 8

        tags = []
        dataset = []
        for sentences in _temp:
            sentences = sentences.lower()
            tempSentence = nltk.wordpunct_tokenize(sentences.lower())
            
            tagSentence = []
            for index, words in enumerate(tempSentence):
                # print(words)
                if "type_person_" in words:
                    tagSentence.append(1)
                    tempSentence[index] = re.sub( "type_person_",'',words)
                elif "type_location_" in words:
                    tagSentence.append(3)
                    tempSentence[index] = re.sub( "type_location_",'',words)
                elif "type_organization_" in words:
                    tagSentence.append(5)
                    tempSentence[index] = re.sub( "type_organization_",'',words)
                elif index>0 and tempSentence[index-1]=="_enamex" and ("type_person_" not in words or "type_location_" not in words or "type_organization_" not in words):
                    tagSentence.append(0)
                elif index>0 and tagSentence[index-1] != 0:
                    if mode=="withIntermediate":
                        if tagSentence[index-1] == 1 or tagSentence[index-1] == 2:
                            tagSentence.append(2)
                        elif tagSentence[index-1] == 3 or tagSentence[index-1] == 4:
                            tagSentence.append(4)
                        elif tagSentence[index-1] == 5 or tagSentence[index-1] == 6:
                            tagSentence.append(6)
                    else:
                        tagSentence.append(tagSentence[index-1])
                elif index==0:
                    tagSentence.append(0)
                else:
                    tagSentence.append(0)

            while True:
                if '' in tempSentence:
                    ind = tempSentence.index('')
                    tempSentence.pop(ind)
                    tagSentence.pop(ind)
                else:
                    break

            while True:
                if '_enamex' in tempSentence:
                    ind = tempSentence.index('_enamex')
                    tempSentence.pop(ind)
                    tagSentence.pop(ind)
                else:
                    break

            tags.append(tagSentence)
            dataset.append(tempSentence)
        return (dataset, tags)


if __name__ == '__main__':
    text = findEntity()
    filtered = text.removeAll()
    file = open("output.txt", "w")
    for sentences in filtered:
        file.write(sentences) 
    file.close()
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import re



stop_words = set(stopwords.words('english'))
ps = PorterStemmer()


def tokenize(text):
    word_tokens = word_tokenize(text)
    lower=[i.lower for i in word_tokens ]
    filtered_sentence = []
    for w in lower:
        if w not in stop_words:
            filtered_sentence.append(w)
    stemmed=[]
    for w in filtered_sentence:
        stemmed.append(ps.stem(w))
            
    return [w for w in stemmed]
def cleaningText(text):   
    preprocesstext1 = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(\d+)"," ",text).split()).strip().lower()
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    preprocesstext2 = pattern.sub(r"\1\1", preprocesstext1)    
    return preprocesstext2    

    
    
    
    


             
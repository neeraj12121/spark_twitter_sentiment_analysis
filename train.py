from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer




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
             
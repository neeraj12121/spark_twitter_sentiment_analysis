from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords




stop_words = set(stopwords.words('english'))


def tokenize(text):
    word_tokens = word_tokenize(text)
    lower=[i.lower for i in word_tokens ]
    filtered_sentence = []
    for w in lower:
        if w not in stop_words:
            filtered_sentence.append(w)
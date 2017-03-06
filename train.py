from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords




stop_words = set(stopwords.words('english'))


def tokenize(text):
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pyspark.mllib.feature import HashingTF, IDF
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import udf
from pyspark.sql.types import *
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

def sentPolarityscore(text):
    ss = sia.polarity_scores(text)
    sent = ss['compound']      
    return float(sent)


def tfidf(doc):
    hashingTF = HashingTF()
    tf = hashingTF.transform(doc)
    tf.cache()
    idf = IDF().fit(tf)
    tf_idf = idf.transform(tf)
    return tf_idf
    
def train():
    udfct = udf(cleaningText,StringType())
    udftokenize = udf(tokenize,ArrayType(StringType))
    df = sqlContext.createDataFrame('tweet.json',('id','text'))
    df.withColumn("id", "text")
    df.PrintSchema()
    
    
    
    
if __name__ == "__main__":
    sc = SparkContext()
    sqlContext = SQLContext(sc)
    sia = SentimentIntensityAnalyzer()




             
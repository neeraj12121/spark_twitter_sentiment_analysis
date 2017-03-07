from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pyspark.mllib.feature import HashingTF, IDF
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, ArrayType, IntegerType, FloatType
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


def sentPolarity(text):
    ss = sia.polarity_scores(text)
    sent = ss['compound']  
    if sent >= 0.0 and sent < 0.20:
        return 0
    elif sent >= 0.20 and sent <= 1.0:
        return 1
    else:
        return -1
    return sent

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
    cleantweet = df.select('id','text',udfct(df.text).alias('ctext'))
    cleantweet.head(1)
    
    sentiscoreUDF = udf(sentPolarityscore,FloatType())
    sentiUDF = udf(sentPolarity,IntegerType()) 


    cleantweet1 = cleantweet.withColumn("sentiscore",sentiscoreUDF(cleantweet.text))
    cleantweet2 = cleantweet1.withColumn("sentiment",sentiUDF(cleantweet.text))
    cleantweet3 = cleantweet2.withColumn("tokens",udftokenize(cleantweet.ctext))

    cleantweet3.printSchema()
    cleantweet3.show(3)

    tweetFinal = cleantweet3.select('id','text','tokens','sentiscore','sentiment')
    tweetFinal.printSchema()


    
    
    
if __name__ == "__main__":
    sc = SparkContext()
    sqlContext = SQLContext(sc)
    sia = SentimentIntensityAnalyzer()




             
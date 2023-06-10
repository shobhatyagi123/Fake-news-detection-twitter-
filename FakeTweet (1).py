# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 11:19:10 2019
@author: ADARSH 
"""
#=====================================================================================================================================
#                                                    PREPROCESSING
#=====================================================================================================================================

# Flask for python server and Cross Origin Resource Sharing (CORS)
from flask import Flask
from flask_cors import CORS

# Tweepy - Python's official Twitter API to extract tweet information & user details
import tweepy 

# NewsAPI - Official newsAPI of 100s of news channel from legitimate sources
from newsapi.newsapi_client import NewsApiClient

# GNnews - Google Search API for verifying from legitimate sources
import json
import requests

# Text Analysis,Cleaning & Ntural Language Processing
import re
import emoji
import string
from rake_nltk import Rake
from textblob import TextBlob
from nltk.corpus import stopwords 
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize

# Machine Learning 
import pandas as pd

# =================================================  SETUP FOR TWEEPY  ===============================================================

consumer_key = "HeN41iSC2AmyFBOW6vX8sKTUs" 
consumer_secret = "MZjHSXATBkLnfZwSI5aoaGN3zCSSh3aCzcKjri3OoWuRggifpO"
access_key = "1010555953647902720-RYonQIwDZ5VmhepIvIrka4fTWzKt5K"
access_secret = "cEN42A0S5f7u8JVW6fS1oXyLfut44rOqXIXplM2oiEmvn"

# Authorization to consumer key and consumer secret 
auth = tweepy.OAuthHandler(consumer_key, consumer_secret) 

# Access to user's access key and access secret 
auth.set_access_token(access_key, access_secret) 

# Calling api 
api = tweepy.API(auth) 

# ============================================  TEXT ANALYSIS FOR NEWS API  ==========================================================

# Count & REMOVE EMOJIS from text

emoji_count = 0                              # Used for checking formality of content later
def give_emoji_free_text(text):
    allchars = [str for str in text]
    emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
    e_count = len(emoji_list)
    clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])
    return (clean_text,e_count)

def newsSentiment(mylist,tweet_senti):
    if len(mylist) < 1:
        return "f"                               # no related news found , return FAKE
    all_senti = list(map(getSentiment,[news for news in mylist]))
    if all_senti.count(tweet_senti)/len(all_senti) >= 0.6 :
        return "hc"    # if all or more than 60% news have the same sentiment as that of tweet
    def getval(num):
        if num > 0 :
            return 1
        elif num < 0 :
            return -1
        return 0
    news_senti = getval(sum(all_senti))
    # above line adds up news sentiments of each article in message_list to find dominating sentiment
    if news_senti == tweet_senti :  
        return "c"                 # most of the news support the tweet so SEEMS CREDIBLE
    else:
        return "sf"                 # most of the news doesn't support the tweet so SEEMS FAKE

def getNewsLabel(NLabel,GLabel):
    if GLabel == NLabel :       # If both agree on same Label
        return GLabel
    elif NLabel == "f":            # ML is confident that news is FAKE
        return "sf"             # No matter if newsAPI says even HC, credibiliy can't be promised
                                                    # RESULT : Seems Fake
    elif NLabel == "sf" or NLabel == "c": 
                                    # ML is neither confident that news is FAKE nor HIGHLY CREDIBLE
         if GLabel == "hc":      # If newsAPI says HC, then might be credible to some extent
             return "c"                          # RESULT : Credible    
         else :                     # If newsAPI says less credible or seems fake
             if NLabel == "c":             
                 return "c"
             return "sf"                        # RESULT : Seems Fake      
    else:                            # ML is confident that news is HIGHLY CREDIBLE no doubt
        return "c"
    
# ===========================================  TEXT ANALYSIS OF TWEET CONTENT  =======================================================
    
def isFormal(sentence):
    err=0               
    if emoji_count > 2:                    # more than 2 emoji count means less formal
        err += emoji_count - 2
    w = ['think','believe','feel'] ; c=0
    wh = ['what','when','where','which','who','whom','why','how']
    puncts = [1 for s in sentence if s in string.punctuation]
    whitesp = [1 for s in sentence if s in string.whitespace]
    totchars =  len(sentence)-len(puncts)-len(whitesp)
    if totchars > 70 :                              # size limited to 70 characters is a concise news
        err+=1
    firstword = sentence.split()[0]              # Most Official news begins with Subject / Noun first
    for x in wn.synsets(firstword):
        if x.pos() == 'n':
            c+=1
        else:
            c-=1
    if c <= 0 :
        err+=1
    if firstword in wh:       # Official news may contain but, never begins with a WH-Question
        err+=1
    if sentence[0] != sentence.capitalize()[0]:   # First letter of first word should be capital
        err+=1
    err += sentence.count("i ")+ sentence.count(" i ")-3   # Not more than 3 times personal pronoun usage
    err += sentence.count("we ")+ sentence.count(" we ")-3
    err += sentence.count("my ")+ sentence.count("our ")-3
    y = [sentence.count(word) for word in w if word in sentence]
    if 'I' in sentence and sum(y):                  # Avoid verbs like 'think', believe', 'feel' with "I"
        err+=sum(y)
    if sentence.count("!") > 2 :                       # atmost 2 or no exclamations & exaggerations
        err+=sentence.count("!")-2
    if err > 4:                                          # Temporary threshold
        return False
    else:
        return True

def getSentiment(text):
    t = TextBlob(text)
    if t.sentiment.polarity > 0:
        return 1
    elif t.sentiment.polarity < 0 :
        return -1
    else:
        return 0

def detectURL(t):
    url = re.findall('https?://t\.co/\S+',t)
    if (len(url) >0):
        return 1
    else:
        return 0
    
# ============================================  TRAINING OF MACHINE LEARNING MODEL  =================================================

# Importing the dataset
dataset = pd.read_csv('TweetSet_100.csv')
X = dataset.iloc[:,3:10]                # remove .values to open the object in variable explorer
y = dataset.iloc[:, 10]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Naive Bayes to the Training set              ACCURACY : 17 correct , 3 incorrect
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()                           

classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

def getFinalLabel(newsLabel,mlLabel):
    d_nums = {"f":0, "sf":1, "c":2, "hc":3}     
    l = {"f":["f","sf","sf","c"],"sf":["sf","sf","sf","c"],"c":["sf","sf","c","c"],"hc":["sf","c","c","hc"]}
    d = {"f":"is Fake", "sf":"seems Fake", "c":"seems Credible", "hc":"is Highly Credible"}
    return d[l[mlLabel][d_nums[newsLabel]]]

#=====================================================================================================================================
#                                              REAL TIME  IMPLEMENTATION 
#=====================================================================================================================================
    
# ============================================ SETTING UP FLASK SERVER ===============================================================

app = Flask(__name__)
CORS(app)
@app.route('/tweet/<id>')
def tweet(id):
       
# ============================= EXTRACTION OF TWEET CONTENT & PARAMETERS USING TWEEPY ================================================
    
    id_of_tweet = id                                # Put Tweet_ID here 
    
    status = api.get_status(id_of_tweet, tweet_mode = 'extended') 
    # if no tweet_mode is specified, gives truncated tweets upto 140 chars only
    
    # Parameters to pass to ML model for analysis 
    
    verified = status.user.verified              # Account verified or not 
    followers = status.user.followers_count      # No. of Followers
    likes = status.favorite_count                # Number of likes
    RT = status.retweet_count                    # Number of Retweets
    tweettext = status.full_text                 # status.text gives truncated tweet wihout tweet_mode
    
# =============================== TWEET CLEANING FOR GENERATING QUERY FOR GNEWS API ==================================================
    
    (text,emoji_count) = give_emoji_free_text(tweettext)
    
    # BASIC CLEANING TEXT 
    
    text2 = re.sub('[^a-zA-Z]',' ',text)
    text2 = text2.lower()
    
    # TOKENIZE into words
    
    word_tokens = word_tokenize(text2)
    
    # REMOVE STOP WORDS like a,an,the,etc....
     
    stop_words = set(stopwords.words('english')) 
    content2 = [w for w in word_tokens if not w in stop_words]
    
    # Find relevant keywords & phrases using RAKE (Rapid Automatic Keyword Extraction)
    
    r = Rake()                        # uses standard stopwards of nltk by default or pass stopwordlist file
    r.extract_keywords_from_text(" ".join(content2))  
    phrases = r.get_ranked_phrases()
    query = phrases[0].split()                # The 1st ranked phrase is most relevant & enough for now
    if len(query) > 8:              #  limited keywords avoid empty results returned
        query = query[:8]
    query = ' '.join(query)

# ================ =======================ANALYSIS OF TWEET CONTENT FOR ML MODEL  ====================================================
    
    formal = isFormal(text)
    sentiment = getSentiment(text)
    URL = detectURL(text)
    
# ===============================NewsAPI & GNEWS API FOR RETRIVAL & GET CLASSIFICATION  ==================================================
    
    #NewsAPI
    newsapi = NewsApiClient(api_key='97220b4495bb442688ef7595889118eb')
    data = newsapi.get_everything(q=query,language='en')  
    articles_array = data['articles']
    message_list=[]
    for article in articles_array:
        message_list.append(article['description'])
    
    # GNews
    url="https://gnews.io/api/v2/"
    token="c7e821d7d236323b18350a160ee68a83"
    response = requests.get(url+'?q='+query+'&token='+token)    # call GNews using query
    results = json.loads(response.text)                         # returns atmost 10 best results
    
    NLabel = newsSentiment(message_list,sentiment)
    GLabel = newsSentiment([x['title'] for x in results['articles']],sentiment)
    
    newsLabel = getNewsLabel(NLabel,GLabel)          # Pass this to final color decider
    
# ============================ ==============MACHINE LEARNING - NAIVE BAYES  =========================================================
    
    # Running the model with the input from extension :
    
    m=["f","sf","c","hc"]
    attribs = ['verified','followers','formal','sentiment','likes','RT','URL']
    x_custom_input = pd.DataFrame([[verified, followers, formal, sentiment, likes, RT, URL]], columns=attribs) 
    x_custom_input = sc.transform(x_custom_input)
    y_custom_input = classifier.predict(x_custom_input)
    
    mlLabel = m[list(y_custom_input)[0]]  # class label to pass to final label decider code
    
# ========================================== FINAL LABEL & COLOR DECISION   ==========================================================
    return "This tweet "+getFinalLabel(newsLabel,mlLabel)

# ============================================ RUN THE FLASK PYTHON SERVER ===========================================================

if __name__ == '__main__':
    app.run()
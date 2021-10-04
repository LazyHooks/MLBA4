# code to preprocess tweets 
import random 
import sys, re
import emoji 
utf8_apostrophe = b'\xe2\x80\x99'.decode("utf8")
string_apostrophe = "'"


def remove_emoji_punc(text):
    """
    removes emojis from text
    """
    
    allchars = [str for str in text]
    emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
    clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])

    
    s1 = clean_text.replace(u'’', u"").replace("'","")
    s1 = re.sub(r'[^a-z0-9 ]+', ' ', s1)
    
    return " ".join(s1.split())
    
    
def separate_hastags_mentions_urls(tweet): 
    """
    returns text and hashtags in a separate list
    removes mentions and URLs
    """
    
    text = tweet.lower()
    hashtag_list = re.findall("#([a-zA-Z0-9_]{1,50})", text)
    
    text = re.sub(r'http\S+', '', text)
    clean_tweet = re.sub("@[A-Za-z0-9_]+","", text)
    clean_tweet = re.sub("#[A-Za-z0-9_]+","", clean_tweet)
    
    return clean_tweet, hashtag_list 

def preprocess_tweet(tweet): 
    """
    1. removes emojis, mentions, punctuations, URLs
    2. returns clean text and hashtags
    """


    clean_tweet, hashtags = separate_hastags_mentions_urls(tweet)
    clean_tweet = remove_emoji_punc(clean_tweet)
    return clean_tweet, hashtags


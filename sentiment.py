# General:
import tweepy       # To consume Twitter's API
import pandas as pd     # To handle data
import numpy as np      # For number computing

# For plotting and visualization:
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns

# We import our access keys:
from credentials import *    # This will allow us to use the keys as variables

from textblob import TextBlob
import re

import csv

# API's setup:
def twitter_setup():
    # Authentication and access using keys:
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

    # Return API with authentication:
    api = tweepy.API(auth)
    return api

# create an extractor object:
extractor = twitter_setup()

tweets = tweepy.Cursor(extractor.search, q="muslim", rpp=20, result_type="recent", include_entities=True, lang="en").items(100)

d = []

for tweet in tweets:
    # create a pandas dataframe as follows:
    d.append({'Date': tweet.created_at,'len':len(tweet.text), 'Tweets': tweet.text.encode('utf-8'), 'ID': tweet.id, 'Likes': tweet.favorite_count, 'RTs': tweet.retweet_count, 'Source': tweet.source})

data = pd.DataFrame(d)
print("Number of tweets extracted: {}.\n".format(len(data)))

# display the first 10 elements of the dataframe:
display(data.head(10))

# extract the mean of lenghts:
mean = np.mean(data['len'])

print("The lenght's average in tweets: {}".format(mean))

# extract the tweet with more FAVs and more RTs:

fav_max = np.max(data['Likes'])
rt_max  = np.max(data['RTs'])

fav = data[data.Likes == fav_max].index[0]
rt  = data[data.RTs == rt_max].index[0]

# Max FAVs:
print("The tweet with more likes is: \n{}".format(data['Tweets'][fav]))
print("Number of likes: {}".format(fav_max))
print("{} characters.\n".format(data['len'][fav]))

# Max RTs:
print("The tweet with more retweets is: \n{}".format(data['Tweets'][rt]))
print("Number of retweets: {}".format(rt_max))
print("{} characters.\n".format(data['len'][rt]))

tfav = pd.Series(data=data['Likes'].values, index=data['Date'])
tret = pd.Series(data=data['RTs'].values, index=data['Date'])


# Likes vs retweets visualization:
tfav.plot(figsize=(16,4), label="Likes", legend=True)
tret.plot(figsize=(16,4), label="Retweets", legend=True)

plt.show()

def clean_tweet(tweet):
    '''
    Utility function to clean the text in a tweet by removing 
    links and special characters using regex.
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def analize_sentiment(tweet):
    '''
    Utility function to classify the polarity of a tweet
    using textblob.
    '''
    analysis = TextBlob(clean_tweet(tweet))
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1

# create a column with the result of the analysis:
data['SA'] = np.array([ analize_sentiment(tweet) for tweet in data['Tweets'] ])

# create a numpy vector mapped to labels:
positive = 0
neutral = 0
negative = 0

for sa in data['SA']:
    if sa == 1:
        positive = positive+1
    elif sa == 0:
        neutral = neutral+1
    else:
        negative = negative +1

size_sa = [positive, neutral, negative]
labels = 'Positive', 'Neutral', 'Negative'
colors = ['green', 'gray', 'red']
explode = (0, 0.1, 0.1)  # explode 1st slice

print ("Total positive sentiment: {}".format(positive))
print ("Total neutral sentiment: {}".format(neutral))
print ("Total negative sentiment: {}".format(negative))

# plt.pie(pie_data, explode=explode, labels=labels, colors=colors,
#         autopct='%1.1f%%')
 
# plt.axis('equal')
# plt.show()


# objects = ('Positive', 'Neutral', 'Negative')
# y_pos = np.arange(len(objects))
# performance = size_sa
 
# plt.bar(y_pos, performance, align='center', alpha=0.5)
# plt.xticks(y_pos, objects)
# plt.ylabel('Number')
# plt.title('Sentiment Analysis')


labels = 'Positive', 'Neutral', 'Negative'
color = ['yellowgreen','gray','lightcoral']
performance = size_sa
explode = (0.1, 0.1, 0.1)
plt.pie(performance,autopct='%1.1f%%',colors=color, labels=labels , shadow= True,explode = explode , startangle=140)
 
plt.show()
# display the updated dataframe with the new column:
# display(data.head(10))
# print(data)


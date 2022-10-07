# from urllib import response
from unicodedata import category
import tweepy
import config
import json

import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

api_key=config.api_key
api_key_secret=config.api_key_secret
bearer_token= config.bearer_token
access_token=config.access_token
access_token_secret=config.access_token_secret

# authentication
auth=tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)

api=tweepy.API(auth) # we can use this to access out twitter account

client = tweepy.Client(bearer_token)

with open("response_sample.json") as f:
    data=json.load(f)

category=[]
for i in range(len(data["productInfoList"])):
    category.append(data["productInfoList"][i]["productBaseInfo"]["productIdentifier"]["categoryPaths"]["categoryPath"][0][0]["title"])
print(category)

product_list=[]
for i in range(len(data["productInfoList"])):
    product_list.append(data["productInfoList"][i]["productBaseInfo"]["productAttributes"]["title"])
print(product_list)

# query="(boat headphones) OR (#boat #headphones) -is:retweet lang:en has:media"
# query="(asus laptop) OR (#asus #laptop) -is:retweet lang:en"
#has:media will limit the tweets but can be used to display photos or videos or gifs

query_list=[]
for i in range(len(category)):
    query_list.append("("+category[i]+" "+product_list[i]+") OR (#"+category[i]+" #"+product_list[i]+") -is:retweet lang:en")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

class HashTable:

	# Create empty bucket list of given size
	def __init__(self, size):
		self.size = size
		self.hash_table = self.create_buckets()

	def create_buckets(self):
		return [[] for _ in range(self.size)]

	# Insert values into hash map
	def set_val(self, key, val):
		# Get the index from the key
		# using hash function
		hashed_key = hash(key) % self.size
		
		# Get the bucket corresponding to index
		bucket = self.hash_table[hashed_key]

		found_key = False
		for index, record in enumerate(bucket):
			record_key, record_val = record
			
			# check if the bucket has same key as
			# the key to be inserted
			if record_key == key:
				found_key = True
				break

		# If the bucket has same key as the key to be inserted,
		# Update the key value
		# Otherwise append the new key-value pair to the bucket
		if found_key:
			bucket[index] = (key, val)
		else:
			bucket.append((key, val))

	# Return searched value with specific key
	def get_val(self, key):
		# Get the index from the key using
		# hash function
		hashed_key = hash(key) % self.size
		
		# Get the bucket corresponding to index
		bucket = self.hash_table[hashed_key]

		found_key = False
		for index, record in enumerate(bucket):
			record_key, record_val = record
			
			# check if the bucket has same key as
			# the key being searched
			if record_key == key:
				found_key = True
				break

		# If the bucket has same key as the key being searched,
		# Return the value found
		# Otherwise indicate there was no record found
		if found_key:
			return record_val
		else:
			return "No record found"

	# Remove a value with specific key
	def delete_val(self, key):
		# Get the index from the key using
		# hash function
		hashed_key = hash(key) % self.size
		
		# Get the bucket corresponding to index
		bucket = self.hash_table[hashed_key]

		found_key = False
		for index, record in enumerate(bucket):
			record_key, record_val = record
			
			# check if the bucket has same key as
			# the key to be deleted
			if record_key == key:
				found_key = True
				break
		if found_key:
			bucket.pop(index)
		return

	# To print the items of hash map
	def __str__(self):
		return "".join(str(item) for item in self.hash_table)

hash_table = HashTable(1000)

rmse=[]#store the rmse for every query

# AUTOMATE THE QUERY:
for query in query_list:
    print(query)

    counts=client.get_recent_tweets_count(query=query, granularity='day')

    tweet_counts=[]
    start_date=[]
    end_date=[]

    for count in counts.data:
        print(count)
        tweet_counts.append(count['tweet_count'])
        start_date.append(count['start'][:10]) #here all dates are string and we are appending it to start_date list...so we can use use slicing on dates for the relevant format
        end_date.append(count['end'][:10]) #only the date is taken into account and not the time
    
    dataset={
        "start_date":start_date,
        # "end_date":end_date,
        "tweet_counts":tweet_counts,
    }
    df=pd.DataFrame(dataset)
    df.index = pd.to_datetime(df['start_date'], format='%Y-%m-%d') #converting start_date to data frame index in order to user the time series model
    del df["start_date"]
    print(df)
    print()

    train = df[df.index <= pd.to_datetime(start_date[len(start_date)-3], format='%Y-%m-%d')]
    test = df[df.index >= pd.to_datetime(start_date[len(start_date)-3], format='%Y-%m-%d')]
    plt.plot(train, color = "black")
    plt.plot(test, color = "red")
    plt.ylabel('Tweet Counts')
    plt.xlabel('Start Date')
    plt.xticks(rotation=45)
    plt.title("Train/Test split for Tweet Data")
    plt.show()

    # Applying SARIMA, ARIMA and ARMA time series model:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    y=train['tweet_counts']
    SARIMAXmodel = SARIMAX(y, order = (1, 0, 1), seasonal_order=(2,2,2,12))
    SARIMAXmodel = SARIMAXmodel.fit()
    y_pred = SARIMAXmodel.get_forecast(len(test.index))
    y_pred_df = y_pred.conf_int(alpha = 0.05)
    y_pred_df["Predictions"] = SARIMAXmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
    y_pred_df.index = test.index
    y_pred_out = y_pred_df["Predictions"] 
    plt.plot(y_pred_out, color='Blue', label = 'SARIMA Predictions')
    plt.legend()

    import numpy as np
    from sklearn.metrics import mean_squared_error
    sarimax_rmse = np.sqrt(mean_squared_error(test["tweet_counts"].values, y_pred_df["Predictions"]))
    print("RMSE: ",sarimax_rmse)

    from statsmodels.tsa.arima.model import ARIMA
    y=train['tweet_counts']
    ARIMAmodel = ARIMA(y, order = (2, 2, 2))
    ARIMAmodel = ARIMAmodel.fit()
    y_pred = ARIMAmodel.get_forecast(len(test.index))
    y_pred_df = y_pred.conf_int(alpha = 0.05) 
    y_pred_df["Predictions"] = ARIMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
    y_pred_df.index = test.index
    y_pred_out = y_pred_df["Predictions"] 
    plt.plot(y_pred_out, color='Yellow', label = 'ARIMA Predictions')
    plt.legend()


    import numpy as np
    from sklearn.metrics import mean_squared_error
    arma_rmse = np.sqrt(mean_squared_error(test["tweet_counts"].values, y_pred_df["Predictions"]))
    print("RMSE: ",arma_rmse)

    from statsmodels.tsa.statespace.sarimax import SARIMAX
    y=train['tweet_counts']
    ARMAmodel = SARIMAX(y, order = (1, 0, 1))
    ARMAmodel = ARMAmodel.fit()
    y_pred = ARMAmodel.get_forecast(len(test.index))
    y_pred_df = y_pred.conf_int(alpha = 0.05) 
    y_pred_df["Predictions"] = ARMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
    y_pred_df.index = test.index
    y_pred_out = y_pred_df["Predictions"]
    plt.plot(y_pred_out, color='green', label = 'Predictions')
    plt.legend()

    import numpy as np
    from sklearn.metrics import mean_squared_error
    arma_rmse = np.sqrt(mean_squared_error(test["tweet_counts"].values, y_pred_df["Predictions"]))
    rmse.append(arma_rmse)
    hash_table.set_val(arma_rmse, query)

rmse.sort()
trending_query=[]
for i in rmse:
    trending_query.append(hash_table.get_val(i))

print(trending_query)




# # search_recent_tweets() function can be used to get tweets relevant to your query from the last seven days.
# c=0
# tweet_id_arr=[] 
# for tweet in tweepy.Paginator(client.search_recent_tweets, query=query, max_results=100).flatten(limit=200):
#     tweet_id_arr.append(tweet.id)
#     c=c+1

# for tweet_id in tweet_id_arr:
#     status = api.get_status(tweet_id, tweet_mode = "extended")
#     full_text = status.full_text
#     print(full_text)

# print(c) #105 equivalent to sum of tweet counts of last 7 days
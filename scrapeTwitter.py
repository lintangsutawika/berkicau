#!/usr/bin/env python
# encoding: utf-8

import tweepy #https://github.com/tweepy/tweepy
import csv
import time
import sys

#Twitter API credentials
consumer_key = sys.argv[1]
consumer_secret = sys.argv[2]
access_key = sys.argv[3]
access_secret = sys.argv[4]

class tweetRetrieval(object):
	"""docstring for tweetRetrieval"""
	def __init__(self, consumer_key, consumer_secret, access_key, access_secret):
			#authorize twitter, initialize tweepy
		self.auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
		self.auth.set_access_token(access_key, access_secret)
		self.api = tweepy.API(self.auth)

	def recursivelyFindFollowers(self, screen_name, recursiveStep=5, maxFollowers=150000):
		ids = []
		initialUser = screen_name
		for items in tweepy.Cursor(api.followers_ids, screen_name=initialUser).items(maxFollowers):
			print(items)
			ids.extend(items)


		for step in (recursiveStep-1):
			_tempID = []
			for userid in ids:
				for items in tweepy.Cursor(api.followers_ids, screen_name=api.get_user(userid).screen_name).items(maxFollowers):
					print(items)
					_tempID.extend(items)
				
			
	def get_all_tweets(self, screen_name, writePath='MoreTweets.tsv', usernamePath='Users.tsv'):
		#Twitter only allows access to a users most recent 3240 tweets with this method
		#initialize a list to hold all the tweepy Tweets
		alltweets = []	
		#make initial request for most recent tweets (200 is the maximum allowed count)
		new_tweets = self.api.user_timeline(screen_name = screen_name,count=200)
		
		#save most recent tweets
		alltweets.extend(new_tweets)
		
		#save the id of the oldest tweet less one
		oldest = alltweets[-1].id - 1
		
		#keep grabbing tweets until there are no tweets left to grab
		while len(new_tweets) > 0:
			print "getting tweets before %s" % (oldest)
			
			#all subsiquent requests use the max_id param to prevent duplicates
			new_tweets = self.api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)
			
			#save most recent tweets
			alltweets.extend(new_tweets)
			
			#update the id of the oldest tweet less one
			oldest = alltweets[-1].id - 1
			
			print "...%s tweets downloaded so far" % (len(alltweets))
		
		with open(usernamePath, 'r') as usernameFile:


		#write the csv
		with open(writePath, 'a') as tweetFile:
			writer = csv.writer(tweetFile)
			outtweets = [[tweet.text.encode("utf-8")] for tweet in alltweets]
			writer.writerows(outtweets)
			tweetFile.close()

		with open(usernamePath, 'a') as usernameFile:
			writer = csv.writer(usernameFile)
			# outtweets = [[tweet.text.encode("utf-8")] for tweet in alltweets]
			writer.writerows([[screen_name]])
			usernameFile.close()
		pass


if __name__ == '__main__':

	tweet = tweetRetrieval(consumer_key, consumer_secret, access_key, access_secret)
	tweet.get_all_tweets("lintangsutawika")

		i = 0
		while(True):
			try:    
				for i in range(i,len(ids)):
					u = api.get_user(ids[i])
					print(u.screen_name)
					get_all_tweets(u.screen_name)
			except:
				i = i + 1
				print("Cannot be pulled")



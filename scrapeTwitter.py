#!/usr/bin/env python
# encoding: utf-8

import tweepy #https://github.com/tweepy/tweepy
import csv
import time
import sys

#Twitter API credentials
username = sys.argv[1]
consumer_key = sys.argv[2]
consumer_secret = sys.argv[3]
access_key = sys.argv[4]
access_secret = sys.argv[5]

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
		
		# with open(usernamePath, 'r') as usernameFile:


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

	userbase = [
	# "gm_gm",
# "temponewsroom",
# "yan_widjaya",
# "Cherly7uno",
# "devinaureel",
# "dhitalarasati",
# "TeukuRyz",
# "yudhakhel",
# "armandmaulana",
# "sherylsheinafia",
# "nabilahJKT48",
# "SachaStevenson",
# "pergijauh",
# "jokoanwar",
# "radityadika",
# "poconggg",
# "EkoSandjojo",
# "ucu_hendarsah",
# "AhmadErani",
# "KemenDesa",
# "jokowi",
# "SBYudhoyono",
# "dillabona",
# "MhdFikri94",
# "okezonenews",
# "mediaIndonesia",
# "antaranews",
# "detikcom",
# "wiranto1947",
# "detiksport",
# "selenamaria_",
# "akmalmarhali ‏",
# "gfnyindonesia ‏",
# "JscmilaUpdate",
# "Toto_B2W",
# "persib",
# "NIVEAMEN_ID",
# "radotvalent",
# "pekopik",
# "ardynshufi",
# "20DETIK",
# "ayo_olahraga",
# "krisnabayu24",
# "Store_Persib",
# "detikfinance",
# "ardian26jp",
# "KamidiaRadisti",
# "Valencia_INA",
# "ramdanilesta",
# "Diahantika",
# "asror4",
# "haditsku",
# "BNI",
# "dedi_ir33",
# "FOXSportsIndo",
# "RHaryantoracing",
# "ayo_olahraga1",
# "AndiePeci",
# "GadisDe22609103",
# "AhokDjarot",
# "addiems",
# "aniesbaswedan",
# "Pak_JK",
# "BeritaJakarta",
# "JSCLounge",
# "BinaMargaDKI",
# "DKIJakarta",
# "Humas_DKI",
# "kpu_dki",
# "JokowiCentre",
# "maswaditya",
# "sherinasinna",
# "bepe20",
# "caitlinhald",
# "PrillyBie",
# "septriasa_acha",
# "Young_Lexx",
# "Arie_Kriting",
# "_NjoyTempe",
# "zarryhendrik",
# "rezaoktovian",
# "Jekibarr",
# "anggika21",
# "ikanatassa",
# "abdurarsyad",
# "indrayr",
# "DMASIV",
# "RaffiAhmadLagi",
"sorayafilms",
"R_AninJKT48",
"leonagustine",
"kaesangp",
"bukalapak"]

	for username in userbase:
		print(username)
		tweet = tweetRetrieval(consumer_key, consumer_secret, access_key, access_secret)
		tweet.get_all_tweets(username)
		ids = []

	# tweepy.Cursor(api.search, q='cricket', geocode="-22.9122,-43.2302,1km").items(10)

	# while(True):
	# 	try:
	# 		for i,items in enumerate(tweepy.Cursor(tweet.api.followers_ids, screen_name=initialUser).items(maxFollowers)):
	# 			print("{}, {}".format(i,items))
	# 			ids.append(items)
	# 			# if i == maxFollowers:
	# 				# break
	# 	except:
	# 		print("waiting 15minutes")
	# 		break
			# time.sleep(60 * 15)

	# i = 0
	# while(True):
	# 	try:    
	# 		for i in range(i,len(ids)):
	# 			u = api.get_user(ids[i])
	# 			print(u.screen_name)
	# 			get_all_tweets(u.screen_name)
	# 	except:
	# 		if i == len(ids):
	# 			break
	# 		else:
	# 			i = i + 1
	# 			print("Cannot be pulled")



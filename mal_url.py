#Nathan Dullea
#Artificial Intelligence Project
#Using Sklearn to Classify Malicious URLs

import pandas as pd
import numpy as np
import sklearn
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

#Function to separate urls into tokens
def makeTokens(input):
	tokenList = []

	#Separate input on slash dash dot and underscore
	tokensBySlash = str(input.encode('utf-8')).split('/')
	for i in tokensBySlash:
		tokensByDot = str(i).split('.')
		for j in tokensByDot:
			tokensByDash = str(j).split('-')
			for k in tokensByDash:
				tokensByUnderscore = str(k).split('_')
				for l in tokensByUnderscore:
					tokenList = tokenList + [l]


	#Turn to set to remove redundancies and back to List
	tokenList = list(set(tokenList))

	#Remove com and com' from tokenList
	if 'com' in tokenList:
		tokenList.remove('com')
	if "com'" in tokenList:
		tokenList.remove("com'")

	return tokenList


def setupDataAndClassifier():
	#Read Data from csv into DataFrame
	data = pd.read_csv('/Users/nathandullea/Desktop/data.csv')

	#Convert to Array and randomize data
	alldata = np.array(data)
	random.shuffle(alldata)

	#Separate Labels and URLs into lists
	y = [d[1] for d in alldata] #labels ('good' or 'bad')
	urls = [d[0] for d in alldata]
	#tfidfvectorizer is equivalent to countvectorizer follower by tfidf transform
	vectorizer = TfidfVectorizer(tokenizer=makeTokens)
	X = vectorizer.fit_transform(urls)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

	#Create Logistic Regression Model
	lgs = LogisticRegression()
	#Train Logistic Regression Model
	lgs.fit(X_train, y_train)
	#Print the Score
	print(lgs.score(X_test, y_test))


setupDataAndClassifier()
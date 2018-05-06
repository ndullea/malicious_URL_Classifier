#Nathan Dullea
#Artificial Intelligence Project
#Using Sklearn to Classify Malicious URLs

#Pandas library is for data analysis, will use it to read csv file
import pandas as pd
import numpy as np
import sklearn
import random
#For exporting trained classifier
#import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

#from sklearn import linear_model
from sklearn.linear_model import LogisticRegression


#import matplotlib.pyplot as plt

#Use Bag of Words approach for tokenizing
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


	#Turn to set to remove redundancies
	tokenList = list(set(tokenList))

	#Remove com and com' from tokenList
	if 'com' in tokenList:
		tokenList.remove('com')
	if "com'" in tokenList:
		tokenList.remove("com'")

	#firstToken = tokenList[:1]
	#print(firstToken)

	return tokenList


def setupDataAndClassifier():
	#Read Data from csv
	data = pd.read_csv('/Users/nathandullea/Desktop/data.csv')
	#Convert to framework (csv already returns framework though?)
	dataFrame = pd.DataFrame(data)

	#Convert to Array and randomize data
	alldata = np.array(dataFrame)
	random.shuffle(alldata)

	#Setup Training and Testing Date
	y = [d[1] for d in alldata]
	urls = [d[0] for d in alldata]
	#tfidfvectorizer is equivalent to countvectorizer follower by tfidf transform
	vectorizer = TfidfVectorizer(tokenizer=makeTokens)
	X = vectorizer.fit_transform(urls)
	#print(X)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	#print(X_train)
	#print(y_train)
	#print(len(urls))
	#print(len(y))


	#Train Logisitc Regression Model
	lgs = LogisticRegression()
	lgs.fit(X_train, y_train)
	print(lgs.score(X_test, y_test))

	#Export to Pickle

	#predict on X_test set
	#y_pred = lgs.predict(X_test)


	#Plot Logistic Regression with matplotlib

	"""
	What am I plotting? Maybe the tfidf score with good/bad on logistic scale?
	"""

	#plt.scatter(X_test, y_test)
	#plt.plot(X_test, y_pred, color='blue', linewidth=5)
	#plt.xticks(())
	#plt.yticks(())
	#plot.scatter()
	#plt.show()
	
	#pyplot.scatter(urls, y)




setupDataAndClassifier()



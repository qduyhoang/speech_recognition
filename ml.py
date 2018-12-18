from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
from sklearn import decomposition, ensemble
from sklearn.feature_selection import SelectPercentile, f_classif

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

import nltk

from sklearn.metrics import precision_recall_fscore_support
import time

import pandas as pd

additional_stop_words = set()

with open('stop-word-list.txt','r') as f:
	line = f.readline()
	while line:
		additional_stop_words.add(line.strip())
		line = f.readline()

#stop words to be used for count vectorizer/tfidf 
stop_words = ENGLISH_STOP_WORDS.union(additional_stop_words)


from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
def stemDoc(doc):
	stemmed_doc = ""
	for word in doc.split():
		stemmed_doc += stemmer.stem(word) + " "
	return stemmed_doc


import os
def iter_files(path):
	"""Walk through all files located under a root path."""
	for dirpath, _, filenames in os.walk(path):
		for f in filenames:
			if f != '.DS_Store':
				yield os.path.join(dirpath, f)

TRAINING_DATA = 'new/'	
POS_feature = False
stemming = True

labels, texts = [], []
for fpath in iter_files(TRAINING_DATA):
	# split_file(fpath)
	with open(fpath, 'r') as f:
		label = os.path.basename(fpath)[:-4]
		labels.append(label)

		
		word_and_pos = ""
		doc = ""
		line = f.readline()
		while line:
			doc += line.strip()
			line=f.readline()

		if stemming:
			doc = stemDoc(doc)

		if POS_feature:
			tokens = nltk.word_tokenize(doc.lower())
			word_and_pos_list = nltk.pos_tag(tokens)
			for word, pos in word_and_pos_list:
				word_and_pos += word + " " + pos + " "
			texts.append(word_and_pos)
		texts.append(doc)


#create a dataframe using texts and labels
trainDF = pandas.DataFrame()
trainDF['text'] = texts
trainDF['label'] = labels


#split the dataset into training and validation datasets
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])

#label encode the target variable
encoder = preprocessing.LabelEncoder()
encoder.fit_transform(trainDF['label'])
train_y = encoder.transform(train_y)
valid_y = encoder.transform(valid_y)

#create a count vectorizer object
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', stop_words=additional_stop_words)
count_vect.fit(trainDF['text'])

#create a tfidf vectorizer object
tfidf_vect = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words=additional_stop_words)
tfidf_vect.fit(trainDF['text'])

#transform the training and validation data using count vectorizer object
xtrain_count = count_vect.transform(train_x)
xvalid_count = count_vect.transform(valid_x)

#transform the training and validation data using count vectorizer object
xtrain_tfidf = tfidf_vect.transform(train_x)
xvalid_tfidf = tfidf_vect.transform(valid_x)

#Select best features
selector = SelectPercentile(f_classif, percentile=20)
# selector.fit(xtrain_tfidf, train_y)

# xtrain_tfidf = selector.transform(xtrain_tfidf)
# xvalid_tfidf = selector.transform(xvalid_tfidf)

# selector.fit(xtrain_count, train_y)

# xtrain_count = selector.transform(xtrain_count)
# xvalid_count = selector.transform(xvalid_count)

# Specify a search grid, then do a 5-fold grid search for each of the feature sets
C_values = [1e-1, 1e1, 1e2, 1.25e2, 1.5e2, 2e2, 1e3]
param_grid_ = {'C': C_values}

# Tune classifier for bag-of-words representation
bow_search = model_selection.GridSearchCV(linear_model.LogisticRegression(), cv=5, 
                                  param_grid=param_grid_)
bow_search.fit(xtrain_count, train_y)

# Tune classifier for tf-idf
tfidf_search = model_selection.GridSearchCV(linear_model.LogisticRegression(), cv=5,
                                   param_grid=param_grid_)
tfidf_search.fit(xtrain_tfidf, train_y)


search_results = pd.DataFrame.from_dict({
                              'bow': bow_search.cv_results_['mean_test_score'],
                              'tfidf': tfidf_search.cv_results_['mean_test_score']},
                              orient='index', 
                              columns= C_values
                      )
print(search_results)


def simple_logistic_classify(X_tr, y_tr, X_test, y_test, description, _C=1):
	### Helper function to train a logistic classifier and score on test data
	m = linear_model.LogisticRegression(C=_C).fit(X_tr, y_tr)
	s = m.score(X_test, y_test)
	print("Test score with", description, "features: ", s)
	return m
m1 = simple_logistic_classify(xtrain_count, train_y, xvalid_count, valid_y, 'bow (before regularization tuning)', )
m2 = simple_logistic_classify(xtrain_tfidf, train_y, xvalid_tfidf, valid_y, 'tfidf (before regularization tuning)')

m1 = simple_logistic_classify(xtrain_count, train_y, xvalid_count, valid_y, 'bow (after regularization tuning)', 
								_C=bow_search.best_params_['C'])
m2 = simple_logistic_classify(xtrain_tfidf, train_y, xvalid_tfidf, valid_y, 'tfidf (after regularization tuning)',
								_C=tfidf_search.best_params_['C'])
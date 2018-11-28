import os

import csv
import pprint
from autocorrect import spell
import enchant
import re

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

import nltk

pp = pprint.PrettyPrinter(indent=4)
additional_stop_words = set()

with open('stop-word-list.txt','r') as f:
	line = f.readline()
	while line:
		additional_stop_words.add(line.strip())
		line = f.readline()

#stop words to be used for count vectorizer/tfidf 
stop_words = ENGLISH_STOP_WORDS.union(additional_stop_words)


def iter_files(path):
	"""Walk through all files located under a root path."""
	for dirpath, _, filenames in os.walk(path):
		for f in filenames:
			if f != '.DS_Store':
				yield os.path.join(dirpath, f)

INPUT_DIR = 'csv/'
OUTPUT_DIR = 'data/'
TRAINING_DATA = 'new/'



"""Generate the dataset"""
def genData(input_dir, output_dir):
	d = enchant.Dict("en_US")
	for fpath in iter_files(input_dir):
		with open(fpath) as in_file, open(OUTPUT_DIR+os.path.basename(fpath)[:-4], 'w') as out_file:
			csv_reader = csv.reader(in_file, delimiter=',')
			i = 1
			for row in csv_reader:
				word_list = ""
				#If the speech column is non-empty
				if len(row[2]):
					for w in row[2].split():
						#spell check and ignore uncorrectable word
						word = spell(w.lower())
						if (d.check(word)):
							if (i % 50 == 0):
								word_list += "\n"
							i += 1
							word_list += word + " "
				out_file.write(word_list)

def split_file(filepath, lines_per_file=20):
	lpf = lines_per_file
	path, filename = os.path.split(filepath)
	with open(filepath, 'r') as f_in:
		basename, ext = os.path.splitext(filename)
		# open the first output file
		file_num = 0
		w = open(os.path.join("%s%s_%0.2d" %(TRAINING_DATA, basename, file_num)), 'w')
		try:
			# loop over all lines in the input file, and number them
			line = f_in.readline()
			i = 0
			while line:
				i += 1
				if (i % lpf == 0):
					file_num += 1
					#possible enhancement: don't check modulo lpf on each pass
					#keep a counter variable, and reset on each checkpoint lpf.
					w.close()
					filename = os.path.join("%s%s_%0.2d" %(TRAINING_DATA, basename, file_num))
					w = open(filename, 'w')

				w.write(line)
				line = f_in.readline()
		finally:
			w.close()

labels, texts = [], []
# genData(INPUT_DIR, OUTPUT_DIR)
#load the dataset
#change parameter to TRAINING_DIR when training
for fpath in iter_files(TRAINING_DATA):
	# split_file(fpath)
	with open(fpath, 'r') as f:
		label = os.path.basename(fpath)[:-3]
		labels.append(label)
		word_and_pos = ""
		doc = ""
		line = f.readline()
		while line:
			doc += line.strip()
			line=f.readline()
		tokens = nltk.word_tokenize(doc.lower())
		word_and_pos_list = nltk.pos_tag(tokens)
		for word, pos in word_and_pos_list:
			word_and_pos += word + " " + pos + " "
		texts.append(word_and_pos)


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
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(trainDF['text'])

#transform the training and validation data using count vectorizer object
xtrain_count = count_vect.transform(train_x)
xvalid_count = count_vect.transform(valid_x)


def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
	#fit the training dataset on the classifier
	classifier.fit(feature_vector_train, label)

	#predict the labels on validation dataset
	predictions = classifier.predict(feature_vector_valid)

	if is_neural_net:
		predictions = predictions.argmax(axis=-1)
	# decoded_labels = encoder.inverse_transform(predictions)
	# decoded_input = tfidf_vect.inverse_transform(feature_vector_valid)
	# for i in range(20):
	# 	print("X=%s, Predicted=%s" % (decoded_input[i], decoded_labels[i]))
	return metrics.accuracy_score(predictions, valid_y)

#Naive Bayes on Count Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
print("Naive Bayes, Count Vectors: ", accuracy)


#Linear classifier on Count Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)
print("LR, Count Vectors: ", accuracy)


	
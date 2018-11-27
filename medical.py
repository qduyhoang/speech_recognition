
import csv
from nltk.tokenize import RegexpTokenizer
import json
from autocorrect import spell
import os




tokenizer = RegexpTokenizer(r'\w+')
digits = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
stop_words = set()
with open('stop-word-list.txt','r') as f:
	line = f.readline()
	while line:
		stop_words.add(line.strip())
		print(stop_words)
		line = f.readline()


INPUT_DIR = 'csv/'
OUTPUT_DIR = 'data/'

def iter_files(path):
	"""Walk through all files located under a root path."""
	for dirpath, _, filenames in os.walk(path):
		for f in filenames:
			if f != '.DS_Store':
				yield os.path.join(dirpath, f)


def genData(filepath):
	with open(filepath) as csv_file, open(OUTPUT_DIR+os.path.basename(filepath), 'w') as output:
		word_tokens = []
		unique_words = {}
		csv_reader = csv.reader(csv_file, delimiter=',')
		for row in csv_reader:
			speech_sent = row[2].lower()
			if len(speech_sent):
				word_tokens.extend(tokenizer.tokenize(speech_sent))
		word_tokens = [w for w in word_tokens if w not in stop_words]
		uniq_tokens = []
		for w in word_tokens:
			#Spellcheck and remove extra space
			revised_token = " ".join(spell(w).split()) 
			if revised_token not in stop_words:
				uniq_tokens.append(revised_token)



		for tok in uniq_tokens:
			if any(i.isdigit() for i in tok) or tok in digits:
				tok = 'number'  
			if tok not in unique_words:
				unique_words[tok] = 1
			else:
				unique_words[tok]+= 1
		unique_words = sorted(unique_words.items(), key=lambda x: x[1], reverse = True)
		for word, count in unique_words:
			output.write('%s , %d\n' %(word, count))

for filepath in iter_files(INPUT_DIR):
	genData(filepath)



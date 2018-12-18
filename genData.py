"""
Generate Data
Step 1: Go through the original CSV dataset. Spellcheck and correct words. Remove uncorrectable ones.
Step 2: Split each file into multiple files based on 'lines_per_file' 
"""
print(__doc__)

import os
import csv
import pprint
from autocorrect import spell
import enchant


def iter_files(path):
	"""Walk through all files located under a root path."""
	for dirpath, _, filenames in os.walk(path):
		for f in filenames:
			if f != '.DS_Store':
				yield os.path.join(dirpath, f)


"""Generate the dataset"""
def genData(input_dir, output_dir):
	d = enchant.Dict("en_US")
	for fpath in iter_files(input_dir):
		with open(fpath) as in_file, open(output_dir+os.path.basename(fpath)[:-4], 'w') as out_file:
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
							if (i % 12 == 0):
								word_list += "\n"
							i += 1
							word_list += word + " "
				out_file.write(word_list)


def split_file(input_dir, output_dir, lines_per_file=3):
	lpf = lines_per_file
	for fpath in iter_files(input_dir):
		path, filename = os.path.split(fpath)

		with open(fpath, 'r') as f_in:
			basename, ext = os.path.splitext(filename)
			# open the first output file
			file_num = 0
			filepath = os.path.join("%s%s_%0.3d" %(output_dir, basename, file_num))
			w = open(filepath, 'w')
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
						filepath = os.path.join("%s%s_%0.3d" %(output_dir, basename, file_num))
						w = open(filepath, 'w')

					w.write(line)
					line = f_in.readline()
			finally:
				w.close()

if __name__ == "__main__":
	import time
	time0 = time.time()
	genData('csv/', 'data/')
	split_file('data/', 'new/', lines_per_file=3)
	time1 = time.time()
	print("Complete generating data after %.3fs" %(time0 - time1))


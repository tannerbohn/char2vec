from __future__ import print_function

from keras.layers import Input, Dense
from keras.models import Model

import numpy as np
import matplotlib.pyplot as plt
import random

def cleanText(text):
	global CHARS
	# lower case

	text = text.strip().lower()
	text = text.replace('_', ' ')

	#text = ''.join(ch for ch in text if ch.isalnum() or ch == ' ')
	#text = ''.join(ch for ch in text if ch in CHARS or ch == ' ')
	newtext = []
	for ch in text:
		if ch in CHARS or ch == ' ':
			newtext.append(ch)
		else:
			newtext.append(' ')
	text = ''.join(newtext)

	text = ' '.join(text.split())
	#text = text.replace(' ', '_')
	return text.strip()

def classToOneHot(item, classes):

	x = [0. for _ in classes]
	
	x[classes.index(item)] = 1.

	return x

def wordToData(word):
	global CHARS

	# x: middle character
	# y: surrounding characters

	X = []
	y = []

	if len(word) < 1: return X, y

	word = '_'+word+'_'

	for i in range(1, len(word)-1):
		#if random.random()< 0.5: continue
		
		#if '_' in word[i-2:i+3]: 
		#	if random.random()< 0.5: continue

		#prev_char2 = word[i-2]
		prev_char = word[i-1]
		mid_char = word[i]
		next_char = word[i+1]
		#next_char2 = word[i+1]

		X.append(classToOneHot(mid_char, CHARS))

		yp=[#classToOneHot(prev_char2, CHARS), 
			classToOneHot(prev_char, CHARS), 
			classToOneHot(next_char, CHARS)] 
			#classToOneHot(next_char2, CHARS)]
		y.append(yp)

	return X, y

def getData(filename):

	f = open(filename, 'r')

	text = f.read()
	f.close()

	text = cleanText(text)

	words = text.split()

	#return words
	
	# map from center word to context (skip-gram)
	X, y = [], []

	# 26622999
	#  5000000
	#for w, i in zip(words[:5000000], xrange(5000000)):
	#	if i%100000 == 0: print(100.*i/5000000)
	for w in words:
		if random.random() < 0.75: continue
		xp, yp = wordToData(w)
		X.extend(np.array(xp, np.float32))
		y.extend(np.array(yp, np.float32))

	X = np.array(X, np.float32) - 0.5
	y = np.array(y, np.float32)# - 0.5

	return X, y
	
	

def findClosest(labels, coords, n=3):
	def dist(p1, p2):
		return np.linalg.norm(np.array(p1)-np.array(p2))
	for label, pos in zip(labels, coords):
		print(label)
		dists = []
		for l2, p2 in zip(labels, coords):
			if l2==label: continue
			dists.append((l2, dist(pos, p2)))
		dists = sorted(dists, key=lambda x : x[1])
		if n == -1:
			if dists[0][1] >= 0.2:
				print("\t{}\t{}".format(dists[0][0], dists[0][1]))
			else:
				for d in dists:
					if d[1] <= 0.15:
						print("\t{}\t{}".format(d[0], d[1]))
		else:		
			for d in dists[:n]:
				print("\t{}\t{}".format(d[0], d[1]))


CHARS = [#'0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
		 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
		 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
		 'u', 'v', 'w', 'x', 'y', 'z', '_']

if __name__ == "__main__":

	print("yo")

	dataFilename = 'data/freud.txt'
	# dataFilename = '/home/tanner/Downloads/internet_archive_scifi_v3.txt'
	# dataFilename = '/home/tanner/Downloads/brown_nolines.txt'

	X, y = getData(dataFilename)

	nb_cols = len(y[0])
	y_cols = [np.array([v[i] for v in y], np.float32) for i in range(nb_cols)]
		
	input_layer = Input(shape=(len(CHARS),))
	encoding = Dense(2, activation='tanh')(input_layer)

	output_1 = Dense(len(CHARS), activation='softmax')(encoding)
	output_2 = Dense(len(CHARS), activation='softmax')(encoding)
	#output_3 = Dense(len(CHARS), activation='softmax')(encoding)
	#output_4 = Dense(len(CHARS), activation='softmax')(encoding)

	model = Model(input=[input_layer], output=[output_1, output_2])#, output_3, output_4])

	encoder = Model(input = [input_layer], output=[encoding])

	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', loss_weights=[0.5, 0.5])

	model.fit(X, y_cols, nb_epoch=15, batch_size=1024, shuffle=True, verbose=True)

	x2 = [classToOneHot(ch, CHARS) for ch in CHARS[:-1]]
	encodings = encoder.predict(np.array(x2))
	for ch, e in zip(CHARS[:-1], encodings):
		print("{}\t{}\t{}".format(ch, e[0], e[1]))


	labels = CHARS[:-1]
	xd = [e[0] for e in encodings]
	yd = [e[1] for e in encodings]
	fig, ax = plt.subplots()
	for i, txt in enumerate(labels):
		_ = ax.annotate(txt, (xd[i],yd[i]))


	findClosest(labels, encodings, n=2)

	plt.show()
	
#!/usr/bin/env python
# coding: utf-8

import string
import numpy as np
import pandas as pd

from gensim.models import Word2Vec


def separateData(dataframe, target, global_cutoff, tt_ratio):
	"""
	Separate datasets into train, val, test.

	We could filter out low frequency samples based on the target and global_cutoff, 
	in which target is one column label in dataframe and global_cutoff is the threshold.

	tt_ratio sets ratio of train/all, test/test+val
	"""
	data = dataframe[dataframe[target].isin(dataframe[target].value_counts().index[dataframe[target].value_counts() > global_cutoff])]
	trainList = []
	for i in data[target].value_counts().index:
		tmpData = data[data[target] == i]
		for j in tmpData.sample(frac=tt_ratio[0], random_state=8853).index.values:
			trainList.append(j)
	train = pd.DataFrame(data.loc[trainList,:]).reset_index(drop=True)
	tmp = pd.DataFrame(data.drop(trainList)).reset_index(drop=True)
	testList = []
	for i in tmp[target].value_counts().index:
		tmpData = tmp[tmp[target] == i]
		for j in tmpData.sample(frac=tt_ratio[1], random_state=8853).index.values:
			testList.append(j)
	test = pd.DataFrame(tmp.loc[testList,:]).reset_index(drop=True)
	val = pd.DataFrame(tmp.drop(testList)).reset_index(drop=True)
	return train, val, test


def convertSeq(seq, maxlen, flank_length):
	"""
	Clean, Upper, Uniform
	Add "N"*flank_length
	RevcompSeq
	"""	
	seq = uniformSeq(upperSeq(cleanSeq(seq)), maxlen)
	return seq + "N"*flank_length + revcompSeq(seq)

def cleanSeq(seq):
	"""
	Remove non-natural bases
	"""
	charList = list(string.printable)
	for i in ["A","C","G","T","a","c","g","t"]:
		charList.remove(i)
	for ch in charList:
		if ch in seq:
			seq = seq.replace(ch, "N")
	return seq

def uniformSeq(seq, maxlen):
	"""
	Uniform sequence to fixed length and add flank_length of N between seq and revcomp
	"""
	if len(seq) > maxlen:
		seq = seq[:maxlen]
	else:
		seq += "N"*(maxlen-len(seq)) 
	return seq

def upperSeq(seq):
	"""
	Convert DNA sequence to uppercase
	"""
	return seq.upper()

def revcompSeq(seq):
	"""
	Get reverse complementary sequence
	"""
	return seq.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def convertSeqW2V(dataframe, column, window, size):
	sentences = []
	for seq in dataframe[column]:
		for shift in range(0, window):
			sentences.append([word for word in re.findall(r".{"+str(window)+"}", seq[shift:])])

	model = Word2Vec(sentences=sentences, size=size, window=4, min_count=1, negative=5, sg=1, sample=0.001, hs=1, 
					 workers=8, seed=8853)

	w2vFeat = []
	for i in range(0, len(sentences), window):
		w2vSum = np.zeros(shape=(size,))
		for j in range(i, i+window):
			for word in sentences[j]:
				w2vSum += model[word]
		w2vFeat.append(w2vSum)
		
	col = ["w2v_{0}".format(i) for i in range(size)]
	tmp = pd.DataFrame(w2vFeat, columns=col)
	return pd.concat([dataframe, tmp], axis=1)
	
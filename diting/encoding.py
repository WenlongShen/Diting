#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd


BaseDict = {"A":[1,0,0,0],
			"C":[0,1,0,0],
			"G":[0,0,1,0],
			"T":[0,0,0,1],
			"N":[0,0,0,0]}

def kmerEncoding2D(seq, kmer, step):
	"""
	first version

	NNNNNN
	---
	kmer
	  ---
	 step	
	"""
	seqNT = []
	for nt in seq[:kmer]:
		seqNT = np.hstack((seqNT, BaseDict.get(nt,[0,0,0,0])))

	for i in range(step, len(seq)+1-kmer, step):
		tmp = []
		for nt in seq[i:i+kmer]:
			tmp = np.hstack((tmp, BaseDict.get(nt,[0,0,0,0])))
		seqNT = np.vstack((seqNT, tmp))
	return np.array(seqNT, dtype=int)

def kmerEncoding3D(seqNT):
	return np.expand_dims(seqNT, 0)


def saveKmerEncodedMat(dataPath, feature_names, target, data, dataName, encoding, params):

	for i,enc in enumerate(encoding):
		tmp = pd.DataFrame()
		tmp[feature_names[i]] = data.apply(lambda df: enc(df["diting_seq"], params[i][0], params[i][1]), axis=1)
		tmp[target] = data[target]
		tmp.to_pickle(dataPath+dataName+"."+feature_names[i]+".pkl")


def loadKmerEncodedMat(dataPath, feature, target, dataName, matType):
	
	tmp = pd.read_pickle(dataPath+dataName+"."+feature+".pkl")
	
	if matType == "3D":
		tmp[feature] = tmp.apply(lambda df: kmerEncoding3D(df[feature]), axis=1)

	inputShape = tmp[feature][0].shape
	numClasses = tmp[target].value_counts().shape[0]

	X_tmp = []
	for i in tmp[feature]:
		X_tmp.append(i)
	X_tmp = np.array(X_tmp)
	y_tmp = pd.get_dummies(tmp[target]).values
	
	return X_tmp, y_tmp, inputShape, numClasses


def motifEncoding(seq, kmer, step):
	return

def histRgbEncoding(seq, kmer, step):
	return

def histStackEncoding(seq, kmer, step):
	return

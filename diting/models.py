#!/usr/bin/env python
# coding: utf-8

from diting.encoding import *

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D
from keras.layers import LSTM
from keras.layers import Dense, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model
from keras import backend as K

from bayes_opt import BayesianOptimization

import subprocess

def model_CNN(dataPath, feature, target):
	
	def model(inputShape, numClasses, n_filters, n_kernel_size, n_units):
		n_filters = int(round(n_filters))
		n_kernel_size = int(round(n_kernel_size))
		n_units = int(round(n_units))
		
		cnnModel = Sequential()
		cnnModel.add(Conv2D(input_shape=inputShape, filters=n_filters, kernel_size=(n_kernel_size, inputShape[-1]), activation="relu", padding ="same", name="conv2d_1", data_format="channels_first"))
		cnnModel.add(MaxPooling2D(pool_size=(int(inputShape[1]/2), 1), strides=None, name="max_pooling2d_1", data_format="channels_first"))
		cnnModel.add(Flatten(name="flatten_1"))
		cnnModel.add(BatchNormalization(name="batch_normalization_1"))
		cnnModel.add(Dense(units=n_units, name="dense_1"))
		cnnModel.add(Activation("relu", name="activation_1"))
		cnnModel.add(BatchNormalization(name="batch_normalization_2"))
		cnnModel.add(Dense(units=numClasses, name="dense_2"))
		cnnModel.add(Activation("softmax", name="activation_2"))
		cnnModel.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

		return cnnModel
	
	def optimize(n_filters, n_kernel_size, n_units, n_epoch, n_batch_size):
		X_train, y_train, inputShape, numClasses = loadKmerEncodedMat(dataPath, feature, target, "train", "3D")
		X_val, y_val, inputShape, numClasses = loadKmerEncodedMat(dataPath, feature, target, "val", "3D")
		
		n_model = model(inputShape, numClasses, n_filters, n_kernel_size, n_units)
		n_epoch = int(round(n_epoch))
		n_batch_size = int(round(n_batch_size))
		
		cnn = n_model.fit(X_train, y_train, batch_size=n_batch_size, validation_data=(X_val, y_val), epochs=n_epoch, verbose=0)
		return cnn.history["val_acc"][-1]
	
	optimizer = BayesianOptimization(
		f = optimize, 
		pbounds = {
			"n_filters":(12, 16),
			"n_kernel_size":(12, 24),
			"n_units":(1, 24),
			"n_epoch":(1,100),
			"n_batch_size":(1,16)
		},
		random_state = 8853
	)
	optimizer.maximize(n_iter = 2)
#	optimizer.maximize(n_iter = 2, alpha = 1e-2, n_restarts_optimizer = 2, acq="ucb", kappa=5)
	print(optimizer.max)

	n_filters = int(round(optimizer.max["params"]["n_filters"]))
	n_kernel_size = int(round(optimizer.max["params"]["n_kernel_size"]))
	n_units = int(round(optimizer.max["params"]["n_units"]))
	n_epoch = int(round(optimizer.max["params"]["n_epoch"]))
	n_batch_size = int(round(optimizer.max["params"]["n_batch_size"]))
	X_train, y_train, inputShape, numClasses = loadKmerEncodedMat(dataPath, feature, target, "train", "3D")
	X_val, y_val, inputShape, numClasses = loadKmerEncodedMat(dataPath, feature, target, "val", "3D")
	
	cnn = model(inputShape, numClasses, n_filters, n_kernel_size, n_units)
	print(cnn.summary())
	cnn.fit(X_train, y_train, batch_size=n_batch_size, validation_data=(X_val, y_val), epochs=n_epoch, verbose=0)
	cnn.save(dataPath+feature+"_cnn.h5")
	plot_model(cnn, to_file=dataPath+feature+"_cnn.png")
	return cnn



def model_LSTM(dataPath, feature, target):
	
	def model(inputShape, numClasses, n_lstmunits, n_units):
		n_lstmunits = int(round(n_lstmunits))
		n_units = int(round(n_units))
		
		lstmModel = Sequential()
		lstmModel.add(LSTM(units=n_lstmunits, input_shape=inputShape, name="lstm_1"))
		lstmModel.add(BatchNormalization(name="batch_normalization_1"))
		lstmModel.add(Dense(units=n_units, name="dense_1"))
		lstmModel.add(Activation("relu", name="activation_1"))
		lstmModel.add(BatchNormalization(name="batch_normalization_2"))
		lstmModel.add(Dense(units=numClasses, name="dense_2"))
		lstmModel.add(Activation("softmax", name="activation_2"))
		lstmModel.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

		return lstmModel
	
	def optimize(n_lstmunits, n_units, n_epoch, n_batch_size):
		X_train, y_train, inputShape, numClasses = loadKmerEncodedMat(dataPath, feature, target, "train", "2D")
		X_val, y_val, inputShape, numClasses = loadKmerEncodedMat(dataPath, feature, target, "val", "2D")
		
		n_model = model(inputShape, numClasses, n_lstmunits, n_units)
		n_epoch = int(round(n_epoch))
		n_batch_size = int(round(n_batch_size))
		
		lstm = n_model.fit(X_train, y_train, batch_size=n_batch_size, validation_data=(X_val, y_val), epochs=n_epoch, verbose=0)
		return lstm.history["val_acc"][-1]
	
	optimizer = BayesianOptimization(
		f = optimize, 
		pbounds = {
			"n_lstmunits":(1, 16),
			"n_units":(1, 16),
			"n_epoch":(1,100),
			"n_batch_size":(1,32)
		},
		random_state = 8853
	)
	optimizer.maximize(n_iter = 2, alpha = 1e-2, n_restarts_optimizer = 2, acq="ucb", kappa=5)
	print(optimizer.max)

	n_lstmunits = int(round(optimizer.max["params"]["n_lstmunits"]))
	n_units = int(round(optimizer.max["params"]["n_units"]))
	n_epoch = int(round(optimizer.max["params"]["n_epoch"]))
	n_batch_size = int(round(optimizer.max["params"]["n_batch_size"]))
	X_train, y_train, inputShape, numClasses = loadKmerEncodedMat(dataPath, feature, target, "train", "2D")
	X_val, y_val, inputShape, numClasses = loadKmerEncodedMat(dataPath, feature, target, "val", "2D")
	
	lstm = model(inputShape, numClasses, n_lstmunits, n_units)
	print(lstm.summary())
	lstm.fit(X_train, y_train, batch_size=n_batch_size, validation_data=(X_val, y_val), epochs=n_epoch, verbose=0)
	lstm.save(dataPath+feature+"_lstm.h5")
	plot_model(lstm, to_file=dataPath+feature+"_lstm.png")
	return lstm



def getEnsembleMat(firstBin, leftBead, totalBin, npArray, beadSize, mat):
	if leftBead == 0:
		mat.append(np.array(npArray))
		return
	for bead in range(leftBead, -beadSize, -beadSize):
		if firstBin == totalBin:
			return
		npArray[firstBin] = bead
		getEnsembleMat(firstBin+1, leftBead-bead, totalBin, npArray, beadSize, mat)
	return


def getBestEnsemble(binNumber, beadNumber, beadSize, preds, y_val):
	npArray = np.zeros(binNumber)
	alphaMat = []
	getEnsembleMat(0, beadNumber, binNumber, npArray, beadSize, alphaMat)
	alphaMat = np.array(alphaMat) / 100

	error = 1
	best = []
	for alpha in alphaMat:
		total = 0
		for i in range(len(alpha)):
			total += alpha[i]*preds[i]
		total = np.argmax(total, axis=1)
		err = np.sum(np.not_equal(total, np.argmax(y_val, axis=1))) / y_val.shape[0]
		if err < error:
			error = err
			best = alpha
	return error, best


def inSilicoMutagenesis(seq, pos):
	seqs = []
	for i in pos:
		tmpSeq = []
		tmpSeq.append(seq[:i] + "A" + seq[i+1:])
		tmpSeq.append(seq[:i] + "C" + seq[i+1:])
		tmpSeq.append(seq[:i] + "G" + seq[i+1:])
		tmpSeq.append(seq[:i] + "T" + seq[i+1:])
		seqs.append(tmpSeq)
	seqs = np.array(seqs)
	return seqs

def getMutationScores(seqs, pos, oriData, model, encoding, params):
	pred = model.predict(oriData)
	mutScores = []
	for o, seq in enumerate(seqs): 
		seqMat = inSilicoMutagenesis(seq, pos)
		target = np.argmax(pred[o])
		oriScore = pred[o][target]
		scores = []
		for i in range(0,seqMat.shape[0]):
			tmpScores = []
			for j in range(0,seqMat.shape[1]):
				tmpMat = kmerEncoding3D(kmerEncoding3D(encoding(seqMat[i][j], params[0], params[1])))
				tmpScores.append(model.predict(tmpMat)[0][target]-oriScore)
			scores.append(tmpScores)
		mutScores.append(scores)
	mutScores = np.array(mutScores)
	return mutScores


def getGradients(model, layerName, data):
	loss = model.get_layer(layerName).output
	grads = K.gradients(loss, [model.input])[0]
	fn = K.function([model.input], [loss, grads])
	return fn([data])

def getSeqImportance(model, layerName, data):
	return getGradients(model, layerName, data)[1]

def getSeqMotifs(model, layerName, data, seqs, dataPath):
	motifs = np.mean(getGradients(model, layerName, data)[0], axis=-1)
	motifs = np.transpose(motifs,(1,0,2))
	motifLen = model.get_layer(layerName).get_weights()[0].shape[0]
	motifLen2 = (motifLen - 1) // 2
	for i, motif in enumerate(motifs):
		fileName = dataPath + layerName + "_motif_" + str(i+1)
		fasta = open(fileName+".fasta", "w")
		for j, pos in enumerate(motif):
			k = np.argmax(pos)
			mot = seqs[j][k-motifLen2:k+motifLen-motifLen2]
			if len(mot) == motifLen:
				fasta.write(">%d_%d_%f\n%s\n" % (j, k, pos[k], mot))
		fasta.close()
		weblogoCmd = 'weblogo -X NO -Y NO --errorbars NO --fineprint "" -C "#008000" A A -C "#0000cc" C C -C "#ffb300" G G -C "#cc0000" T T < %s.fasta > %s.eps' % (fileName, fileName)
		subprocess.call(weblogoCmd, shell=True)

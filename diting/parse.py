#!/usr/bin/env python
# coding: utf-8

from diting.utils import *

import pandas as pd
from pandas.io.json import json_normalize
from Bio import SeqIO


def parse_addgene(file_name, columns, maxlen, flank_length):
	"""
	The json file provided by Addgene.
	"""
	data = json_normalize(pd.read_json(file_name)["plasmids"])
	data["diting_seq"] = data.apply(GetAddgeneSeq, axis=1, maxlen=maxlen, flank_length=flank_length)
	data["pi"] = data.apply(lambda df: convertAddgenePi(df["pi"]), axis=1)
	columns.append("diting_seq")
	return data[columns]

def getAddgeneSeq(data, maxlen, flank_length):
	seqs = ""
	if len(data["sequences.public_addgene_full_sequences"]) > 0:
		for seq in data["sequences.public_addgene_full_sequences"]:
			seqs += (seq["sequence"] + "N"*flank_length)
	elif len(data["sequences.public_addgene_partial_sequences"]) > 0:
		for seq in data["sequences.public_addgene_partial_sequences"]:
			seqs += (seq["sequence"] + "N"*flank_length)
	else:
		for seq in data["sequences.public_user_full_sequences"]:
			seqs += (seq["sequence"] + "N"*flank_length)
		for seq in data["sequences.public_user_partial_sequences"]:
			seqs += (seq["sequence"] + "N"*flank_length)
	return convertSeq(seqs, maxlen, flank_length)

def convertAddgenePi(data):
	x = 0
	if len(data) > 0:
		x = data[0]
	return str(x)


def parse_fasta(file_name, y, maxlen, flank_length):
	meta=[]
	sequence=[]
	for seq in SeqIO.parse(file_name, "fasta"):
		meta.append(str(seq.id))
		sequence.append(convertSeq(str(seq.seq), maxlen, flank_length))
	return pd.DataFrame(data={"meta":meta, "diting_seq":sequence, "type":y})

#!/usr/bin/env python
# coding: utf-8

from diting.models import *

from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
import matplotlib.patches as patches

import matplotlib.pyplot as plt
import seaborn as sns

colorScheme = {"A": "#008000", "C": "#0000cc", "G": "#ffb300", "T": "#cc0000", "N": "#333333"}

def drawNT(ax, text, x=0.0, y=0.0, width=1.0, height=1.0, color="#008000", edgecolor="None", font = FontProperties(family="monospace")):
	tp = TextPath((0.0, 0.0), text, size=1, prop=font)
	bbox = tp.get_extents()
	bwidth = bbox.x1-bbox.x0
	bheight = bbox.y1-bbox.y0
	txt = Affine2D()
	txt.translate(-bbox.x0, -bbox.y0)
	txt.scale(1/bwidth*width, 1/bheight*height)
	txt.translate(x, y)
	tp = tp.transformed(txt)
	patch = patches.PathPatch(tp, facecolor=color, edgecolor=edgecolor)
	ax.add_patch(patch)

def drawMotif(ax, motif, ylim):
	for i, heights in enumerate(motif):
		nt_pos = sorted(zip(heights, "ACGT"))
		nt_neg = sorted(zip(heights, "ACGT"), reverse=True)
		y = 0
		for height, nt in nt_pos:
			if height > 0:
				drawNT(ax, nt, x=0.5+i, y=y, height=height, color=colorScheme[nt])
				y += height
		y = 0
		for height, nt in nt_neg:
			if height < 0:
				drawNT(ax, nt, x=0.5+i, y=y, height=height, color=colorScheme[nt])
				y += height
	ax.set_xticks(list(range(motif.shape[0]+1)))
	ax.set_ylim(ylim[0], ylim[1])

def heatMotif(ax, motif, ylim):
	sns.heatmap(motif.transpose(), cmap='RdBu_r', linewidths=0.2, vmin=ylim[0], vmax=ylim[1])
	ax = plt.gca()
#	ax.set_xticks(list(range(motif.transpose().shape[1])))
	ax.set_xticklabels(range(1, motif.transpose().shape[1]+1))
	ax.set_yticklabels("ACGT", rotation="horizontal")

def plot_CNNfilters(model, layerName, plotType, ylim):
	fig = plt.figure(figsize=(16, 9))
	fig.subplots_adjust(hspace=1, wspace=1)
	filters = model.get_layer(layerName).get_weights()[0]
	#transpose for plotting
	filters = np.transpose(filters,(3,0,1,2)).squeeze(axis=-1)
	plotNum = int(len(filters)**0.5) + 1
	for i, nfilter in enumerate(filters):
		ax = fig.add_subplot(plotNum, plotNum, i+1)
		if plotType == "text":
			drawMotif(ax, nfilter, ylim)
		elif plotType == "heatmap":
			heatMotif(ax, nfilter, ylim)
		ax.set_title("Filter %s" % (str(i+1)))
		ax.autoscale_view()
	return fig

def plot_Sequences(sequences, site, plotType, ylim):
	fig = plt.figure(figsize=(16, 9))
	fig.subplots_adjust(hspace=0.3, wspace=1)
	sequences = sequences.squeeze(axis=1)
	plotNum = len(sequences)
	for i, sequence in enumerate(sequences):
		ax = fig.add_subplot(plotNum, 1, i+1)
		if plotType == "text":
			drawMotif(ax, sequence[site,:], ylim)
		elif plotType == "heatmap":
			heatMotif(ax, sequence[site,:], ylim)
		ax.set_title("Sequence %s" % (str(i+1)))
		ax.autoscale_view()
	return fig

def plot_Mutations(sequences, site, plotType, ylim):
	fig = plt.figure(figsize=(16, 9))
	fig.subplots_adjust(hspace=0.3, wspace=1)
	plotNum = len(sequences)
	for i, sequence in enumerate(sequences):
		ax = fig.add_subplot(plotNum, 1, i+1)
		if plotType == "text":
			drawMotif(ax, sequence[site,:], ylim)
		elif plotType == "heatmap":
			heatMotif(ax, sequence[site,:], ylim)
		ax.set_title("Sequence %s" % (str(i+1)))
		ax.autoscale_view()
	return fig

import sys
import numpy as np
import os,sys,inspect
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches as patches
from collections import defaultdict
import time
import seaborn as sns

# make sure SOM is in PATH
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0,current_dir)
from SOMs.SOM import *

def color_test():
	"""
	Simple SOM Showcase using rgb-color-vectors
	"""
	
	map_dim = (30,20)
	
	# data
	values = np.random.randint(0, 256, (2000, 3)).astype(np.float64) #colors
	decrease = "linear"
	# decrease = "exp"
	# using RGBA - Vectors for simplicity 
	values /= np.linalg.norm(values, axis=1).reshape(values.shape[0], 1)
	
	# som setup
	# for colors periodic_boundarys = False is simply prettier
	# PCA give the option to use PCA for initial Neuron distribution (WARNING: this can cause neurons to have values outside of [0,1], 
	# 																	if not enough epochs are run afterwards)
	# pool_size only important, if train_async is used
	h = lambda x2,sgma: np.exp(-x2/(2*sgma)) # guass
	som = SOM(map_dim,len(values[0]),values,PCA=True,periodic_boundarys=False,neighbourhood_function=h)
	
	# Training 
	# learning_rate gives initial learning rate, while learning_rate_end gives learning rate in last epoch
	# lr_decrease gives the function connecting both: "linear" for linear and "exp" for exponential
	# same goes for sigma, aka radius
	start = time.time()

	#som.train(sigma=6,learning_rate = 0.2,learning_rate_end = 0.001,
	#		max_epochs = 50000,sigma_end = 1, radius_decrease = "exp", 
	#		lr_decrease = "exp")

	print("Training time:",time.time() - start,"s")

	# unused test-values
	# get mapped as example, but wont be used
	test_values = np.random.randint(0, 256, (5000, 3)).astype(np.float64)
	test_values /=  np.linalg.norm(test_values, axis=1).reshape(test_values.shape[0], 1)
	out = som.map(test_values)
	
	# Plot
	fig = plt.figure()
	# setup axes
	ax = fig.add_subplot(111, aspect='equal')
	ax.set_xlim((0, map_dim[0]))
	ax.set_ylim((0, map_dim[1]))
	
	# decide if you wnat to plot neurons or plot the mapped data. Neurons is defentily prettier
	show_neurons = True
	
	if show_neurons:
		for i,neuron in enumerate(som.weights):
			ax.add_patch(patches.Rectangle(som.Grid[i], 1, 1, facecolor=neuron, edgecolor='none'))
	else:
		for result,color in zip(out,test_values):
			ax.add_patch(patches.Rectangle(result, 1, 1, facecolor=color, edgecolor='none'))
	plt.xlabel("x")
	plt.ylabel("y")
	plt.show()
	
	
	# show umatrix
	sns.heatmap(som.get_umatrix())
	plt.xlabel("x")
	plt.ylabel("y")
	plt.title("U-Matrix")
	plt.show()
	
	# show component planes
	for comp in range(3):
		component_plane = som.weights[:,comp].reshape((map_dim))
		sns.heatmap(component_plane,linewidth = 0,rasterized=False,cmap=["Reds","Greens","Blues"][comp])
		plt.xlabel("x")
		plt.ylabel("y")
		plt.title("Distribution of %s"%["red","green","blue"][comp])
		plt.show()
	
if __name__ == "__main__":
	color_test()

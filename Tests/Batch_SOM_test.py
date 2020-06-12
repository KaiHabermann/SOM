import sys
import numpy as np
import os,sys,inspect
import pandas as pd
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0,current_dir)
print(parent_dir)
from SOM_neu import *
import matplotlib.pyplot as plt
from matplotlib import patches as patches
from collections import defaultdict
from scipy.spatial import ConvexHull
import seaborn as sn
import time


def color_test():
	map_dim = (30,20)
	values = np.random.randint(0, 256, (2000, 3)).astype(np.float64) #colors
	# print(values)
	decrease = "linear"
	values /= np.linalg.norm(values, axis=1).reshape(values.shape[0], 1)
	som = batch_SOM(map_dim,3,values,max_epochs=500,batch_size=100,lerning_rate = 0.2,neighbourhood_function=gauss(0.5*map_dim[0]),sigma_decay=1.,lerning_rate_decay = 0.01,PCA=True,radius_decrease = decrease, lr_decrease = decrease,pool_size=2,periodic_boundarys=False,sigma=3)
	print(som.indim)
	start = time.time()
	som.train_c(prnt = False)
	print(time.time() - start)
	start = time.process_time()
	# som.train(prnt = False)
	print(time.process_time() - start)
	start = time.process_time()
	#som.train(prnt = False)
	print(time.process_time() - start)

	# som.train(prnt = True)
	
	print(np.max(som.weights))
	test_values = np.random.randint(0, 256, (5000, 3)).astype(np.float64)
	test_values /=  np.linalg.norm(test_values, axis=1).reshape(test_values.shape[0], 1)
	out = som.map(test_values)
	# Plot
	fig = plt.figure()
	# setup axes
	ax = fig.add_subplot(111, aspect='equal')
	ax.set_xlim((0, map_dim[0]))
	ax.set_ylim((0, map_dim[1]))
	
	show_neurons = True
	
	if show_neurons:
		for i,neuron in enumerate(som.weights):
			ax.add_patch(patches.Rectangle(som.Grid[i], 1, 1, facecolor=neuron, edgecolor='none'))
	else:
		for result,color in zip(out,test_values):
			ax.add_patch(patches.Rectangle(result, 1, 1, facecolor=color, edgecolor='none'))
	
	
	for comp in range(3):
		component_plane = som.weights[:,comp].reshape((map_dim))
		sn.heatmap(component_plane,linewidth = 0,rasterized=False,cmap=["Reds","Greens","Blues"][comp])
		plt.xlabel("x")
		plt.ylabel("y")
		plt.title("Verteilung von %s"%["rot","grün","blau"][comp])
		plt.show()

	
	
		

if __name__ == "__main__":
	color_test()
	
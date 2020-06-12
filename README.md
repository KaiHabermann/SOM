# SOM_for_ATLAS

## Setup:
- setup by installing git lfs<br/>

- then run sh setup.sh <br/>

- then run git pull a final time to pull all large data files<br/>

## Exampe code for 60x90 SOM:

```python
	
	from SOM import *
	import numpy as np
	import seborn as sns
	import matplotlib.pyplot as plt
	
	map_dim = (60,90)
	values = np.loadtxt("csv_files/2lep_complete.csv",delimiter=",",skiprows=1)
	decrease = "linear"
	# decrease = "exp"
	
	# som setup
	# PCA give the option to use PCA for initial Neuron distribution
	# pool_size only important, if train_async is used
	som = batch_SOM(map_dim,len(values[0]),values,PCA=False,periodic_boundarys=True)
	
	# no training required, when we load an existing som
	som.load("csv_files/trainierte_soms/60x90.csv")
	
	sns.heatmap(som.get_umatrix())
	plt.ylabel("y")
	plt.xlabel("x")
	plt.title("U-Matrix")
	plt.show()
	
	mapped_values = som.map(values)
	hit_histogram = np.zeros(map_dim)
	print(mapped_values)
	for x,y in mapped_values:
		hit_histogram[x][y] += 1
	
	sns.heatmap(hit_histogram)
	plt.ylabel("y")
	plt.xlabel("x")
	plt.title("Heatmap der Treffer")
	plt.show()
	
	
```
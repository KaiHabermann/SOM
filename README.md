# SOM_for_ATLAS

## Setup:
- setup by installing git lfs<br/>

- then run sh setup.sh <br/>

- then run git pull<br/>

## Exampe code for 60x90 SOM:

```python
	
	from SOM import *
	import numpy as np
	import seborn as sns
	import matplotlib.pyplot as plt
	
	values = np.loadtxt("csv_files/2lep_complete.csv",delimiter=",",skiprows=1)
	decrease = "linear"
	# decrease = "exp"
	values /= np.linalg.norm(values, axis=1).reshape(values.shape[0], 1)
	
	# som setup
	# PCA give the option to use PCA for initial Neuron distribution
	# pool_size only important, if train_async is used
	som = batch_SOM(map_dim,len(values[0]),values,PCA=False,periodic_boundarys=True)
	som.load("csv_files/trainierte_soms/60x90.csv")
	
	sns.heatmap(som.get_umatrix())
	plt.shpw()
	
```
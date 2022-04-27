# SOM

## Build:
- sh build.sh will build the projects cython and c files<br/>

- there is no installation script yet<br/>

- you need to alter you PYTHONPATH manually<br/>

## Use:
- SOM class is much more flexible and can accept any neighbourhood function depending on r^2 and a parameter sigma

- Batch_SOM is less felxible, but much faster. Here we train in Batches, meaning, that the update is only performed after an entire batch has been presented. For this class the user can trigger parallel training, but this only is effective for large batches. Large batches are not allways advisable.
For non parallel training here a dedicated c library is used. This forces the user to use gaussian distances though. To use this speed up for non batched data, one can simply st the batch size to 1 and have a normal training.

- For both forms of training, we can have either linear or exponential decrease of the learning rate and sigma. 
For how this is done look into the test files in the Tests/ folder

## Exampe code for 60x90 SOM:

```python
	
from SOM import SOM, batch_SOM
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


# Training 
# learning_rate gives initial learning rate, while learning_rate_end gives learning rate in last epoch
# lr_decrease gives the function connecting both: "linear" for linear and "exp" for exponential
# same goes for sigma, aka radius
som.train(prnt = True,batch_size=500,learning_rate = 1,sigma_end=1.,learning_rate_end = 0.0001,sigma=8,
	radius_decrease = decrease, lr_decrease = decrease,max_epochs=5000)


# no training required, when we load an existing som
#som.load("csv_files/trainierte_soms/60x90.csv")

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

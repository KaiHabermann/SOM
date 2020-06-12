import numpy as np 
import random
import json
from collections import defaultdict
from multiprocessing import Pool, Array, get_context
import ctypes
import global_arrays
from scipy.stats import f, norm,levene, bartlett, mannwhitneyu

global_arrays.data = [None,None]
try:
	_c_extension = ctypes.CDLL('/Users/kai/Desktop/Bachelorarbeit/OSM/libsom.so')
	_c_extension.train_from_c.argtypes = (ctypes.c_int , ctypes.c_int, ctypes.c_int,ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,ctypes.c_int, ctypes.c_int, ctypes.c_int)
	_c_extension.train_from_c.restype = ctypes.POINTER(ctypes.c_double)

	_c_extension.train_from_c_periodic.argtypes = (ctypes.c_int , ctypes.c_int, ctypes.c_int,ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,ctypes.c_int, ctypes.c_int, ctypes.c_int)
	_c_extension.train_from_c_periodic.restype = ctypes.POINTER(ctypes.c_double)

	_c_extension.map_from_c.argtypes = (ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)
	_c_extension.map_from_c.restype = ctypes.POINTER(ctypes.c_int)

	C_INIT_SUCESS = True
except:
	C_INIT_SUCESS = False

def gauss(sgma):
	def f(x,sgma = sgma):
		return np.exp(-(x)/(2*sgma**2))
	return f


def e_func(k):
	return lambda x: np.exp(-k*abs(x))
def eins(x):
	return x
	
def winning_neuron_async_c(args):
	pass
	# call c function

def winning_neuron_asyinc_large_batch(args):
		in_vals,lr,sgma,rad,periodic, outdim = args
		outdim = np.asarray(outdim)
		W = global_arrays.data[0] 
		G = global_arrays.data[1] 
		new_W = np.zeros(W.shape,dtype=np.float64)
		new_W_temp = np.zeros(W.shape,dtype=np.float64)

		new_w_divisor = np.zeros(W.shape[:-1])[:, np.newaxis]
		for x in in_vals:
			index_flat = np.argmin(np.sum((x-W)**2,axis=1))
			new_W_temp[index_flat] += x
			new_w_divisor[index_flat] += 1
		
		bool_index = (new_w_divisor != 0).flatten()
		new_W_temp[bool_index]/=new_w_divisor[bool_index]
		new_w_divisor *= 0
		
		for g,x in zip(G[bool_index],new_W_temp[bool_index]):
			if periodic:
				delta = np.abs(G - g) 
				delta = np.where(delta > 0.5 * outdim, delta - outdim, delta) # decide on wicht way to go
				d = np.sum(delta**2,axis=1)
			else:
				d = np.sum((G-g)**2,axis=1)
		
			limit = d < rad**2
			# d = np.sum(np.square(self.Grid - self.Grid[index_flat]), axis=1)
			# Topological Neighbourhood Function
			h =  np.exp(-(d[limit])/(2*sgma**2))[:, np.newaxis] # np.roll(self.distance,index_flat)[:,np.newaxis]#
			new_W[limit] += lr * h*(x - W[limit]) 
			new_w_divisor[limit] += h

		
		return [new_W,new_w_divisor]



def winning_neuron_asyinc(args):
	in_vals,lr,sgma,rad,periodic, outdim = args
	outdim = np.asarray(outdim)
	W = global_arrays.data[0] 
	G = global_arrays.data[1] 
	new_W = np.zeros(W.shape)
	new_w_divisor = np.zeros(W.shape[:-1])[:, np.newaxis]
	a = []
	
	for x in in_vals:
		a.append(np.argmin(np.sum((x-W)**2,axis=1)))
	if rad > 1:
		
		for index_flat,x in zip(a,in_vals):
			g = G[index_flat]
						
			### no periodic boundry ###
			# ne.evaluate('sum((G-g)**2,1)',out=self.d) 
			if periodic:
				delta = np.abs(G - g) 
				delta = np.where(delta > 0.5 * outdim, delta - outdim, delta) # decide on wicht way to go
				d = np.sum(delta**2,axis=1)
			else:
				d = np.sum((G-g)**2,axis=1)
		
			limit = d < rad**2
			# Topological Neighbourhood Function
			h =  np.exp(-(d[limit])/(2*sgma**2))[:, np.newaxis] # np.roll(self.distance,index_flat)[:,np.newaxis]#
			new_W[limit] += lr * h*(x - W[limit]) 
			new_w_divisor[limit] += h
	else:
		# radius is too small for updaates out of the bmu
		
		for index_flat,x in zip(a,in_vals):							
			### no periodic boundry ###
		
			# Topological Neighbourhood Function
			new_W[index_flat] += lr * (x - W[index_flat]) 
			new_w_divisor[index_flat] += 1.

	return [new_W,new_w_divisor]
	
	
class SOM(object):
	def __init__(self,outdim,indim,trainings_set,neighbourhood_function = gauss(2),sigma=2,lerning_rate = 0.2,lerning_rate_decay = 1000,max_epochs = 10000,sigma_decay = 100,PCA = False,validate = None,break_condition = None,keys = None,periodic_boundarys = False,random=False, radius_decrease = "exp", lr_decrease = "exp"):
		
			
		self.time_const = lerning_rate_decay
		self.sigma_time_const = sigma_decay
		self.tr_set = trainings_set
		self.h = neighbourhood_function
		self.Grid = np.mgrid[0:outdim[0],0:outdim[1]].reshape(2,outdim[0]*outdim[1]).T # mapping flat index to map postition
		self.max_epochs = max_epochs
		self.sigma = sigma
		self.lerning_rate = lerning_rate
		self.initail_lerning_rate = lerning_rate
		self.initial_sigma = sigma
		self.periodic = periodic_boundarys
		self.random = random
		
		self.radius_decrease = radius_decrease
		self.lr_decrease = lr_decrease
		self.outdim = np.asarray(outdim)
		self.indim = indim
		self.validation_data = validate
		if break_condition is not None:
			self.break_point = break_condition
			self.keys = keys
			self.keyrange = np.max(keys) + 1

		if PCA:
			self.PCA_preprocessing()
		else:
			self.weights = self.tr_set[np.random.randint(0,len(self.tr_set),outdim[0]*outdim[1])].copy()
			#self.weights /= np.linalg.norm(self.weights, axis=1).reshape(self.weights.shape[0], 1)

		
		# helpers to reduce memory work
		# self.d = np.zeros(self.weights.shape[0],dtype=np.int64)
		
		# lookup array for distances
		# self.distance = np.zeros(outdim[0] * outdim[1] )

			
		
	def PCA_preprocessing(self):
		from sklearn.decomposition import PCA
		from sklearn.preprocessing import QuantileTransformer
		self.pca = PCA(n_components=2)
		
		
		from sklearn.preprocessing import StandardScaler
		x = StandardScaler().fit_transform(self.tr_set)
		
		#getting principle components of all input vectors
		QT = QuantileTransformer()
		
		principalComponents = self.pca.fit_transform(x)
		
		QT.fit(principalComponents)
		
		
		max_PC = np.array([np.max(principalComponents[:,0]),np.max(principalComponents[:,1])])
		min_PC = np.array([np.min(principalComponents[:,0]),np.min(principalComponents[:,1])])
		
		
		
		def transform_sapce(x):
			#print( (max_PC - min_PC ) /float(self.outdim))
			#print(x)
			x *=  (max_PC - min_PC ) /self.outdim
			x +=  min_PC
			return x
		
		self.weights = np.zeros((self.outdim[0]*self.outdim[1], self.indim))
		
		for position in self.Grid:
			index =transform_sapce(position.astype(np.float64)) # get map postition in pca-space
			vec = self.pca.inverse_transform(QT.inverse_transform([index,]))
			self.weights[int(position[0])*self.outdim[1] + int(position[1])] = vec
		print('PCA weights initialized. Variance Ratio: %s'%self.pca.explained_variance_ratio_)
								
		
		
	def winning_neuron(self,x, W):
		# Also called as Best Matching Neuron/Best Matching Unit (BMU)
		return np.argmin(np.sum((x-W)**2,axis=1))

	def update_weights(self,lr, x, W):
		i = self.winning_neuron(x, W)
		g = self.Grid[i]
		G = self.Grid
		if self.periodic:
			### periodic boundry ###
			delta = np.abs(G - g) 
			delta = np.where(delta > 0.5 * self.outdim, delta - self.outdim, delta) # decide on wicht way to go
			self.d = np.sum(delta**2,axis=1)
		
		else:
			### no periodic boundry ###
			self.d = np.sum((G-g)**2,axis=1) 
		# d = np.sum(np.square(self.Grid - self.Grid[index_flat]), axis=1)
		# Topological Neighbourhood Function
		h = lr * self.h(self.d)[:, np.newaxis]
		# ne.evaluate('W + h * (x - W)',out=W)
		W+=h*(x - W)
		return W

	def decay_learning_rate(self,eta_initial, epoch, time_const):
		if self.lr_decrease == "exp":
			return eta_initial * np.exp(-epoch/time_const)			
		elif self.lr_decrease == "linear":
			return eta_initial  - (eta_initial - 0.001)/self.max_epochs  * epoch

	def decay_variance(self,sigma_initial, epoch, time_const):
		if self.radius_decrease == "exp": 
			return gauss(sigma_initial * np.exp(-epoch/time_const))
		
			
	def decay_variance_async(self,sigma_initial, epoch, time_const):
		if self.radius_decrease == "exp":
			return sigma_initial * np.exp(-epoch/time_const)
		if self.radius_decrease == "linear":
			return sigma_initial - (sigma_initial - 1) *epoch/self.max_epochs

	def set_tr_set(self,trainings_set):
		self.tr_set = trainings_set
	
	def train(self):
		epoch = 0
		while epoch <= self.max_epochs:
			element = self.tr_set[random.randint(0,len(self.tr_set)-1)]
			self.weights = self.update_weights(self.lerning_rate,element,self.weights)
			epoch += 1
			self.lerning_rate = self.decay_learning_rate(self.initail_lerning_rate,epoch,self.time_const)
			self.decay_variance(self.initial_sigma,epoch,self.sigma_time_const)
			
			

	def map(self,input_values):
		return [self.Grid[self.winning_neuron(x,self.weights)] for x in input_values]
	
	def map_array_flat(self,input_values):
		result = np.zeros(len(input_values),dtype=np.int)
		for i,x in enumerate(input_values):
			result[i] = self.winning_neuron(x, self.weights)
		return result

	def map_seccond_best_array_flat(self,input_values):
		result = np.zeros(len(input_values),dtype=np.int)
		for i,x in enumerate(input_values):
			result[i] = np.argpartition(np.sum((self.weights-x)**2,axis=1), 2)[2] # get seccond smallest element
		return result
		
	def filter_for_bbox(self,input_values,bbox):
		bbox = np.asarray(bbox)
		mapped_points = np.asarray(self.map(input_values))
		mask = np.logical_and(np.all(mapped_points >= bbox[0],axis= 1) ,np.all(mapped_points <= bbox[1],axis=1))
		return input_values[mask]
		
	
		
	def quantization_error(self,data):
		results = self.map_array_flat(data)
		distances = np.sum((self.weights[results] - data)**2,axis=1)**0.5
		return np.sum(distances)/len(distances)
	
	def map_embedding_accuracy(self,data,confidence = 0.05):
		################ mean in confidence interval ##########################
		
		z = norm.ppf(1.-confidence)
						
		mean_neurons = np.sum(self.weights,axis=0)/len(self.weights)
		mean_data = np.sum(data,axis=0)/len(data)
		
		var_neurons = np.var(self.weights,axis=0)
		var_data = np.var(data,axis=0)
		
		
		
		variance_multiplicator = (var_neurons**2/len(self.weights) + var_data**2/len(data))**0.5
		is_in_interval_mean = np.asarray([mannwhitneyu(a,b,alternative='two-sided')[1] > confidence for a,b in zip(data.transpose(),self.weights.transpose())]).astype(np.bool)
		################ variance in confidence interval ##########################
		
		
		is_in_interval_var = np.asarray([levene(a,b,center='trimmed')[1] > confidence for a,b in zip(data.transpose(),self.weights.transpose())]).astype(np.bool) # if p_values < conficende than variances are not the same
		
		
		################ result #########################################
		is_embedded = np.logical_and((~is_in_interval_mean) , (~is_in_interval_var)).astype(np.float64)
		
		return np.sum(is_embedded)/len(is_embedded)
		
		
	def topographic_error(self,data):
		x_0 = self.Grid[self.map_array_flat(data)]
		x_1 = self.Grid[self.map_seccond_best_array_flat(data)]
		
		is_neighbour = np.any(abs(x_0-x_1) > 1,axis = 1).astype(np.float64)
		
		return np.sum(is_neighbour)/len(is_neighbour)
		
	 	
		
	def topological_error(self,data):
		raise(NotImplementedError)
		
		
		
	def get_umatrix(self):
		umatrix = np.zeros(tuple(self.outdim.tolist()))
		weights = self.weights.reshape((self.outdim[0],self.outdim[1],self.indim))
		for i in range(self.outdim[0]):
			for j in range(self.outdim[1]):
				if i > 0:
					umatrix[i][j] += np.sum((weights[i][j] - weights[i-1][j])**2)**0.5
				if i < self.outdim[0] -1:
					umatrix[i][j] += np.sum((weights[i][j] - weights[i+1][j])**2)**0.5
				if j > 0:
					umatrix[i][j] += np.sum((weights[i][j] - weights[i][j-1])**2)**0.5
				if j < self.outdim[1] -1:
					umatrix[i][j] += np.sum((weights[i][j] - weights[i][j+1])**2)**0.5
		return umatrix
		
	
	def topopgraphic_product(self,k_end = -1):
		if k_end == -1:
			k_end = self.weights.shape[0] - 1
		elif k_end > self.weights.shape[0] - 1:
			k_end = self.weights.shape[0] - 1
		
		topo_map = np.ones(self.weights.shape[0])
		outdim = np.asarray(self.outdim)
		if k_end > len(self.weights) - 1:
			k_end = len(self.weights) - 1
		for i,element in enumerate(self.weights):
			
			if self.periodic:
				delta = abs(self.Grid - self.Grid[i]) 
				delta = np.where(delta > 0.5 * outdim, delta - outdim, delta) # decide on wicht way to go
				d_out = np.sum(delta**2,axis=1)**0.5
				knn_output_whole = np.argsort(d_out)
			else:
				d_out = np.sum((self.Grid-self.Grid[i])**2,axis=1)**0.5
				knn_output_whole = np.argsort(d_out)

			knn_input_whole = np.argsort(np.sum((self.weights-element)**2,axis=1)) # 

			
			for k in range(2, k_end):
				knn_output = knn_output_whole[1:k]
				knn_input = knn_input_whole[1:k] # get k smallest elements, each element might me in the mix twice
				
				if self.periodic:
					delta = abs(self.Grid[knn_output] - self.Grid[i]) 
					delta = np.where(delta > 0.5 * outdim, delta - outdim, delta) # decide on wicht way to go
					d = np.sum(delta**2,axis=1)**0.5
				else:
					d = np.sum((self.Grid[knn_output]-self.Grid[i])**2,axis=1)**0.5
				
				if self.periodic:
					delta = abs(self.Grid[knn_input] - self.Grid[i])
					delta = np.where(delta > 0.5 * outdim, delta - outdim, delta) # decide on wicht way to go
					d_knn_input = np.sum(delta**2,axis=1)**0.5
				else:
					d_knn_input = np.sum((self.Grid[knn_input] - self.Grid[i])**2, axis=1)**0.5
					
				p1_d = np.sum((self.weights[knn_output] - element)**2, axis=1)**0.5
				d_temp = (np.sum((self.weights[knn_input] - element)**2, axis=1))**0.5
				
				p1 =  np.nansum( np.log((p1_d)  /  d_temp )) 
				
				p2 =  np.nansum( np.log(d_knn_input / d ))
				
								
				topo_map[i] +=( p1 +  p2 )*(1/(2*k))
		
		return  np.sum(topo_map)/(self.weights.shape[0] * (k_end) ), topo_map.reshape(tuple(self.outdim.tolist()))
			
	
	def k_means_input(self,k=3,cluster_start=None,return_object = False):
		
		"""
		Perform clusetering of neurons in input space
		"""
		
		from pyclustering.cluster.kmeans import kmeans
		from pyclustering.cluster.center_initializer import random_center_initializer
		from pyclustering.utils.metric import type_metric, distance_metric
		
		
					
		if cluster_start is None:
			centers_initial = random_center_initializer(self.weights, k).initialize()  	
		else :
			centers_initial = []
			for point in cluster_start:
				centers_initial.append(self.weights[np.all(self.Grid == point,axis=1)][0])
		k_means = kmeans(self.weights,centers_initial)
		k_means.process()
		
		
		self.cluster_ = np.zeros(len(self.Grid))
		for cluster_num, indices in enumerate(k_means.get_clusters()):
			self.cluster_[np.array(indices)] = cluster_num
		
		if return_object: 
			# want to recieve entire object?
			return k_means
		return self.cluster_

		
		
			
	def save(self,filename):
		if '.json' in filename:
			with open(filename, "w") as f:
				json.dump(self.weights.tolist(), f)
		elif '.csv' in filename:
			np.savetxt(filename, self.weights, delimiter=',')
				
	def load(self, filename):
		if '.json' in filename:
			with open(filename, "r") as f:
				self.weights = np.asarray(json.load(f))
		elif '.csv' in filename:
			self.weights = np.loadtxt(filename,delimiter=",")
			
	
			
class batch_SOM(SOM):
	def __init__(self,outdim,indim,trainings_set,neighbourhood_function = gauss(2),sigma=2,lerning_rate = 0.2,lerning_rate_decay = 1000,max_epochs = 100,sigma_decay = 100,batch_size=1000,pool_size = 36 ,PCA=False,validate = None,break_condition = None,keys = None,periodic_boundarys=False,random=False,radius_decrease = "exp", lr_decrease = "exp"):
		
		super().__init__(outdim,indim,trainings_set,neighbourhood_function = neighbourhood_function,sigma=sigma,lerning_rate = lerning_rate,lerning_rate_decay = lerning_rate_decay,max_epochs = max_epochs,sigma_decay = sigma_decay,PCA=PCA,validate = validate,break_condition = break_condition,keys = keys,periodic_boundarys=periodic_boundarys,random=random,radius_decrease=radius_decrease,lr_decrease=lr_decrease)
		
		self.batch_size = batch_size
		self.pool_size = pool_size 
		self.trained_c = False
	
	def map_c(self,values):
		global _c_extension
		c_pointer = ctypes.POINTER(ctypes.c_double)
		input_values_pp = (c_pointer * len(values)) ()
		
		for i,a in enumerate(values):
			input_values_pp[i] = (ctypes.c_double * len(a))()
			for j in range(len(a)):
				input_values_pp[i][j] = a[j]
		
		c_x = ctypes.c_int(self.outdim[0])
		c_y = ctypes.c_int(self.outdim[1])
		c_input_dim = ctypes.c_int(self.indim)
		c_input_size = ctypes.c_int(len(values))

		mapped_values_c = _c_extension.map_from_c(self.weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),input_values_pp,c_x,c_y,c_input_dim,c_input_size)
		mapped_values = np.ctypeslib.as_array(mapped_values_c,shape=(len(values),2))
		return mapped_values
	
	def train_async(self,prnt=False):
		epoch = 0
		while epoch < self.max_epochs:
			if self.batch_size < len(self.tr_set):
				epoch_iter = epoch % (len(self.tr_set) // self.batch_size)
				if self.random:
					elements = self.tr_set[np.random.randint(0,len(self.tr_set),self.batch_size)]
				else:
					elements = self.tr_set[epoch_iter*self.batch_size: (epoch_iter+1)*self.batch_size]
			else:
				elements = self.tr_set

			self.weights += self.update_weights_async(self.lerning_rate, elements, self.weights) 
			# prepare next epoch
			epoch += 1
			self.lerning_rate = self.decay_learning_rate(self.initail_lerning_rate,epoch,self.time_const)
			self.sigma = self.decay_variance_async(self.initial_sigma,epoch,self.sigma_time_const)
			if prnt:
				print(epoch)


	def _train_c(self,prnt=False):
		
		global _c_extension
		c_pointer = ctypes.POINTER(ctypes.c_double)
		input_values_pp = (c_pointer * len(self.tr_set)) ()
		
		for i,a in enumerate(self.tr_set):
			input_values_pp[i] = (ctypes.c_double * len(a))()
			for j in range(len(a)):
				input_values_pp[i][j] = a[j]
		
		### convert python to c types
		c_x = ctypes.c_int(self.outdim[0])
		c_y = ctypes.c_int(self.outdim[1])
		c_input_dim = ctypes.c_int(self.indim)
		c_input_size = ctypes.c_int(len(self.tr_set))
		c_learning_rate = ctypes.c_double(self.lerning_rate)
		c_sigma = ctypes.c_double(self.sigma)
		c_learning_rate_end = ctypes.c_double(self.time_const)
		c_sigma_end = ctypes.c_double(self.sigma_time_const)
		c_linear = ctypes.c_int(1 if self.lr_decrease == "linear" else 0)
		c_batchsize = ctypes.c_int(self.batch_size)
		c_epochs = ctypes.c_int(self.max_epochs)
		c_prnt =  ctypes.c_int(prnt)
		
		if self.periodic:
			self.weights_c = _c_extension.train_from_c_periodic(c_x,c_y,c_input_dim,input_values_pp,c_input_size,c_learning_rate,c_sigma,c_learning_rate_end,c_sigma_end,c_linear,c_batchsize,c_epochs,c_prnt)
		else:
			self.weights_c = _c_extension.train_from_c(c_x,c_y,c_input_dim,input_values_pp,c_input_size,c_learning_rate,c_sigma,c_learning_rate_end,c_sigma_end,c_linear,c_batchsize,c_epochs,c_prnt)
		self.weights = np.ctypeslib.as_array(self.weights_c, shape=(self.outdim[0]*self.outdim[1],self.indim))
		self.trained_c = True
		
	def _train(self,prnt=False):
		epoch = 0
		while epoch < self.max_epochs:
			elements = self.tr_set[np.random.randint(0,len(self.tr_set)-1,self.batch_size)]
			self.weights = self.update_weights(self.lerning_rate, elements, self.weights) 
			# prepare next epoch
			epoch += 1
			self.lerning_rate = self.decay_learning_rate(self.initail_lerning_rate,epoch,self.time_const)
			self.sigma = self.decay_variance_async(self.initial_sigma,epoch,self.sigma_time_const)
			#print(epoch)
			if prnt:
				print(epoch)

		
	def train(self,prnt=False):
		if C_INIT_SUCESS:
			self._train_c(prnt)
		else:
			self._train(prnt)
	
		
		
	def update_weights_async(self,lr, elements, W):
		f = winning_neuron_asyinc if self.batch_size < self.outdim[0] * self.outdim[1] else winning_neuron_asyinc_large_batch
		
		pool_sze = self.pool_size
		#print(len(shaerd_weights))
		global_arrays.data[0] = W
		global_arrays.data[1] = self.Grid
		with Pool(processes=pool_sze) as pool:
			rad = abs(2*self.sigma**2 * np.log(1e-10))**0.5
			iterations = [(elements[i*len(elements)//pool_sze:(i+1) * len(elements)//pool_sze],self.lerning_rate,self.sigma,rad,self.periodic,self.outdim) for i in range(pool_sze)]
			new_maps = pool.map(f,iterations)
		
		
		for new_map in new_maps[1:]:
			new_maps[0][0] += new_map[0]
			new_maps[0][1] += new_map[1]	
		
		new_maps[0][0][(new_maps[0][1]==0).flatten()] *= 0
		new_maps[0][1][new_maps[0][1]==0] = 1
		return new_maps[0][0]/new_maps[0][1]


	def update_weights(self,lr, elements, W):
		winners = [self.winning_neuron(x, W) for x in elements]
		G = self.Grid
		for index_flat,x in zip(winners,elements):
			g = self.Grid[index_flat]
			
			if self.periodic:
				### periodic boundry ###
				delta = np.abs(G - g) 
				delta = np.where(delta > 0.5 * self.outdim, delta - self.outdim, delta) # decide on wicht way to go
				self.d = np.sum(delta**2,axis=1)
			
			else:
				### no periodic boundry ###
				self.d = np.sum((G-g)**2,axis=1) 
			
			
			# d = np.sum(np.square(self.Grid - self.Grid[index_flat]), axis=1)
			# Topological Neighbourhood Function
			h = lr * np.exp(-(self.d)/(2*self.sigma**2))[:, np.newaxis]  # np.roll(self.distance,index_flat)[:,np.newaxis]
			W += h*(x - W)
		return W
		
			
		
		
		
	

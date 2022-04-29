import numpy as np 
import random
import json
from collections import defaultdict
from multiprocessing import Pool, Array, get_context
import ctypes
from c_init import _c_extension, C_INIT_SUCESS
import global_arrays
from scipy.stats import f, norm,levene, mannwhitneyu

def gauss(d2,sgma):
	return np.exp(-(d2)/(2*sgma**2))

def e_func(k):
	return lambda x: np.exp(-k*abs(x))
def eins(x):
	return x
	
def winning_neuron_async_c(args):
	raise NotImplementedError("Async c accelerated training not yet available")


class SOM(object):
	def __init__(self,outdim,indim,trainings_set,PCA = False,periodic_boundarys = False,random=False,neighbourhood_function = gauss):
		"""
		outdim: the dimension of the output
		indim: the dimension of the input
		training_set: the training data shape = (indim,Ndata)
		periodic_boundarys: use periodic boundarys?
		random: choose training data randomly? If False order of training set will be used. If Ndata < Ntraining the system will loop arround.
		neighbourhood function: f(r^2,sigma) the distance function taking 2 parameters
		"""
				
		self.tr_set = trainings_set
		self.h = neighbourhood_function
		self.Grid = np.mgrid[0:outdim[0],0:outdim[1]].reshape(2,outdim[0]*outdim[1]).T # mapping flat index to map postition
		self.periodic = periodic_boundarys
		self.random = random
		
		self.outdim = np.asarray(outdim)
		self.indim = indim
		self.weights_initialized = False
		if PCA:
			self.PCA_preprocessing()
			self.weights_initialized = True
		else:
			self.weights = self.tr_set[np.random.randint(0,len(self.tr_set),outdim[0]*outdim[1])].copy()
				
		
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

	def _update_weights(self,lr, x, W,sigma):
		i = self.winning_neuron(x, W)
		g = self.Grid[i]
		G = self.Grid
		if self.periodic:
			### periodic boundry ###
			delta = np.abs(G - g) 
			delta = np.where(delta > 0.5 * self.outdim, delta - self.outdim, delta) # decide on wicht way to go
			d = np.sum(delta**2,axis=1)
		
		else:
			### no periodic boundry ###
			d = np.sum((G-g)**2,axis=1) 
		# Topological Neighbourhood Function
		h = lr * self.h(d,sigma)[:, np.newaxis]
		W+=h*(x - W)
		return W

	def decay_learning_rate(self,eta_initial, epoch, time_const):
		if self.lr_decrease == "exp":
			return eta_initial * np.exp(-epoch/time_const)			
		elif self.lr_decrease == "linear":
			return eta_initial  - (eta_initial - 0.001)/self.max_epochs  * epoch

	def decay_variance(self,sigma_initial, epoch, time_const):
		if self.radius_decrease == "exp": 
			return sigma_initial * np.exp(-epoch/time_const)
		
			
	def decay_variance_async(self,sigma_initial, epoch, time_const):
		if self.radius_decrease == "exp":
			return sigma_initial * np.exp(-epoch/time_const)
		if self.radius_decrease == "linear":
			return sigma_initial - (sigma_initial - 1) *epoch/self.max_epochs

	def set_tr_set(self,trainings_set):
		self.tr_set = trainings_set
	
	def train(self,sigma=2,learning_rate = 0.2,learning_rate_end = 0.001,
			max_epochs = 10000,sigma_end = 1, radius_decrease = "exp", 
			lr_decrease = "exp"):
		
		# set all parameters before training
		self.lr_end = learning_rate_end
		self.sigma_end = sigma_end
		
		if radius_decrease ==  "linear":
			self.sigma_time_const = ( (learning_rate - learning_rate_end))/max_epochs
		elif radius_decrease == "exp":
			self.sigma_time_const = (-1.)*max_epochs  /(np.log(learning_rate_end) - np.log(learning_rate))
			
		if lr_decrease ==  "linear":
			self.time_const = ((sigma - sigma_end))/max_epochs
		elif lr_decrease == "exp":
			self.time_const = (-1.)*max_epochs  /(np.log(sigma_end) - np.log(sigma))
		
		self.radius_decrease = radius_decrease
		self.lr_decrease = lr_decrease
		self.max_epochs = max_epochs
		self.sigma = sigma
		self.learning_rate = learning_rate
		self.initail_learning_rate = learning_rate
		self.initial_sigma = sigma

		epoch = 0
		for epoch in range(self.max_epochs):
			element = random.choice(self.tr_set)
			self.weights = self._update_weights(self.learning_rate,element,self.weights,sigma)
			self.learning_rate = self.decay_learning_rate(self.initail_learning_rate,epoch,self.time_const)
			sigma = self.decay_variance(self.initial_sigma,epoch,self.sigma_time_const)
		
	def _map_c(self,values):
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
		try:
			c_weights = self.weights_c
		except:
			c_weights = self.weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
		
		mapped_values_c = _c_extension.map_from_c(c_weights,input_values_pp,c_x,c_y,c_input_dim,c_input_size)
		mapped_values = np.ctypeslib.as_array(mapped_values_c,shape=(len(values),2))
		return mapped_values

	def _activation_matrix_c(self,values):
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
		try:
			c_weights = self.weights_c
		except:
			c_weights = self.weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
		
		mapped_values_c = _c_extension.activation_from_c(c_weights,input_values_pp,c_x,c_y,c_input_dim,c_input_size)
		mapped_values = np.ctypeslib.as_array(mapped_values_c,shape=(self.outdim))
		return mapped_values
	

	def map(self,input_values):
		if C_INIT_SUCESS:
			return self._map_c(input_values)
		else:
			return [self.Grid[self.winning_neuron(x,self.weights)] for x in input_values]
			
	def activation_matrix(self,input_values):
		if C_INIT_SUCESS:
			return self._activation_matrix_c(input_values)
		else:
			raise(NotImplementedError("Only if C-Backend if available"))
		
	
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
		
	def filter_for_bbox(self,input_values,points):
		# wants points as a list of [(x,y,z,...), ]
		# all values mapped to these points will be returned
		points = np.asarray(points).transpose()
		mapped_points = np.asarray(self.map(input_values))
		mask = np.zeros(len(input_values),dtype=bool)
		for i,point in enumerate(mapped_points):
			#print(points==point)
			if np.any(np.all(points==point,axis=1)):
				mask[i] = True
		#mask = np.logical_and(np.all(mapped_points >= bbox[0],axis= 1) ,np.all(mapped_points <= bbox[1],axis=1))
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
		"""
		returns universal distance matrix for trained SOM
		"""
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
		
		"""
		topographic product as described by poelzbauer
		k_end ist for processing time, since processing time increases with O(n**4) 
		k_end gives depth of iteration for each neuron
		"""
		
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
				
				# nans may happen
				p1 =  np.nansum( np.log((p1_d)  /  d_temp )) 
				
				p2 =  np.nansum( np.log(d_knn_input / d ))
				
				topo_map[i] +=( p1 +  p2 )*(1/(2*k))
		
		return  np.sum(topo_map)/(self.weights.shape[0] * (k_end) ), topo_map.reshape(tuple(self.outdim.tolist()))
			
	
	def k_means_input(self,k=3,cluster_start=None,return_object = False):
		
		"""
		Perform clusetering of neurons in input space
		cluster_start gives starting points for alogrithm
		return_object gives k_means instance instead of labels for the mapped elements
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
		else:
			raise(ValueError("File needs to be .json or .csv"))
				
	def load(self, filename):
		if '.json' in filename:
			with open(filename, "r") as f:
				self.weights = np.asarray(json.load(f))
		elif '.csv' in filename:
			self.weights = np.loadtxt(filename,delimiter=",")
		else:
			raise(ValueError("File needs to be .json or .csv"))

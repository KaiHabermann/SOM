import numpy as np 
from multiprocessing import Pool
import ctypes
from SOMs.SOM import SOM, gauss
from helpers.c_init import _c_extension, C_INIT_SUCESS


def winning_neuron_asyinc_large_batch(args):
		in_vals,lr,sgma,rad,periodic, outdim,G,W = args
		outdim = np.asarray(outdim)

		# new_W is the updated W for after the batch
		# the temp version is needed, because we are in multiprocess
		new_W = np.zeros(W.shape,dtype=np.float64)
		hitting_data = np.zeros(W.shape,dtype=np.float64)

		new_w_divisor = np.zeros(W.shape[:-1])[:, np.newaxis]
		for x in in_vals:
			# do update for hit neuron 
			# we now have hitting_data[i] = sum(hitting data points on neuron i)
			# we can use this to speed up training in case of batchsiize > N_neurons
			index_flat = np.argmin(np.sum((x-W)**2,axis=1))
			hitting_data[index_flat] += x
			new_w_divisor[index_flat] += 1
		
		bool_index = (new_w_divisor != 0).flatten()
		hitting_data[bool_index]/=new_w_divisor[bool_index]
		new_w_divisor[bool_index] = 0
		
		for g,x in zip(G[bool_index],hitting_data[bool_index]):
			# h is the same for all datappoints mapped to the same neuron
			# so we first add the datapoints mapped to aneuron (see above)
			# and then iterate over these summed savlues
			if periodic:
				delta = np.abs(G - g) 
				delta = np.where(delta > 0.5 * outdim, delta - outdim, delta) # decide on wicht way to go
				d = np.sum(delta**2,axis=1)
			else:
				d = np.sum((G-g)**2,axis=1)

			# this value limits the influence a hit has on its sourrounding
			limit = d < rad 
			# d = np.sum(np.square(self.Grid - self.Grid[index_flat]), axis=1)
			# Topological Neighbourhood Function
			# functions can not be pickeld, so we can not use a user defined function here
			h =  np.exp(-(d[limit])/(2*sgma**2))[:, np.newaxis] # np.roll(self.distance,index_flat)[:,np.newaxis]#
			new_W[limit] += lr * h*(x - W[limit]) 
			new_w_divisor[limit] += h

		return [new_W,new_w_divisor]

def winning_neuron_asyinc(args):
	# standard batch som algorithm
	in_vals,lr,sgma,rad,periodic, outdim,G,W = args
	outdim = np.asarray(outdim)
	new_W = np.zeros(W.shape)
	new_w_divisor = np.zeros(W.shape[:-1])[:, np.newaxis]
	a = []
	
	for x in in_vals:
		a.append(np.argmin(np.sum((x-W)**2,axis=1)))
	if rad > 1: # min distance is 1 
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
		# radius is too small for updates out of the bmu
		for index_flat,x in zip(a,in_vals):							
			### no periodic boundry ###
		
			# Topological Neighbourhood Function
			new_W[index_flat] += lr * (x - W[index_flat]) 
			new_w_divisor[index_flat] += 1.

	return [new_W,new_w_divisor]

class batch_SOM(SOM):
	def __init__(self,outdim,indim,trainings_set,pool_size = 2 ,PCA=False,periodic_boundarys=False,random=False):
		
		super().__init__(outdim,indim,trainings_set,neighbourhood_function = None,PCA=PCA,periodic_boundarys=periodic_boundarys,random=random)
		self.pool_size = pool_size 
		self.trained_c = False
		
	def train_async(self,prnt=False,sigma=2,learning_rate = 0.2,learning_rate_end = 0.001,max_epochs = 10000,batch_size = 1000,sigma_end = 1, radius_decrease = "exp", lr_decrease = "exp"):
			
		# set all parameters before training
		self.lr_end = learning_rate_end
		self.sigma_end = sigma_end
		self.lr_decrease = lr_decrease
		self.radius_decrease = radius_decrease
		
		if radius_decrease ==  "linear":
			self.sigma_time_const = ( (learning_rate - learning_rate_end))/max_epochs
		elif radius_decrease == "exp":
			self.sigma_time_const = (-1.)*max_epochs  /(np.log(learning_rate_end) - np.log(learning_rate))
			
		if lr_decrease ==  "linear":
			self.time_const = ((sigma - sigma_end))/max_epochs
		elif lr_decrease == "exp":
			self.time_const = (-1.)*max_epochs  /(np.log(sigma_end) - np.log(sigma))
		
		self.max_epochs = max_epochs
		self.sigma = sigma
		self.learning_rate = learning_rate
		self.initail_learning_rate = learning_rate
		self.initial_sigma = sigma
		self.batch_size = batch_size

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

			self.weights += self._update_weights_async(self.learning_rate, elements, self.weights) 
			# prepare next epoch
			epoch += 1
			self.learning_rate = self.decay_learning_rate(self.initail_learning_rate,epoch,self.time_const)
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
		c_learning_rate = ctypes.c_double(self.learning_rate)
		c_sigma = ctypes.c_double(self.sigma)
		c_learning_rate_end = ctypes.c_double(self.lr_end)
		c_sigma_end = ctypes.c_double(self.sigma_end)
		c_linear_rad = ctypes.c_int(1 if self.radius_decrease == "linear" else 0)
		c_linear_lr = ctypes.c_int(1 if self.lr_decrease == "linear" else 0)
		c_initial_weights = self.weights.copy().ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if self.weights_initialized else ctypes.POINTER(ctypes.c_double)()

		c_batchsize = ctypes.c_int(self.batch_size)
		c_epochs = ctypes.c_int(self.max_epochs)
		c_prnt =  ctypes.c_int(prnt)
		
		if self.periodic:
			self.weights_c = _c_extension.train_from_c_periodic(c_x,c_y,c_input_dim,input_values_pp,c_initial_weights,c_input_size,c_learning_rate,c_sigma,c_learning_rate_end,c_sigma_end,c_linear_rad,c_linear_lr,c_batchsize,c_epochs,c_prnt)
		else:
			self.weights_c = _c_extension.train_from_c(c_x,c_y,c_input_dim,input_values_pp,c_initial_weights,c_input_size,c_learning_rate,c_sigma,c_learning_rate_end,c_sigma_end,c_linear_rad,c_linear_lr,c_batchsize,c_epochs,c_prnt)
		self.weights = np.ctypeslib.as_array(self.weights_c, shape=(self.outdim[0]*self.outdim[1],self.indim))
		self.trained_c = True

	def _train(self,prnt=False):
		epoch = 0
		while epoch < self.max_epochs:
			elements = self.tr_set[np.random.randint(0,len(self.tr_set)-1,self.batch_size)]
			self.weights = self._update_weights(self.learning_rate, elements, self.weights) 
			# prepare next epoch
			epoch += 1
			self.learning_rate = self.decay_learning_rate(self.initail_learning_rate,epoch,self.time_const)
			self.sigma = self.decay_variance_async(self.initial_sigma,epoch,self.sigma_time_const)
			#print(epoch)
			if prnt:
				print(epoch)
		
	def train(self,prnt=False,sigma=2,learning_rate = 0.2,learning_rate_end = 0.001,max_epochs = 10000,batch_size = 1000,sigma_end = 1, radius_decrease = "exp", lr_decrease = "exp"):
		
		# set all parameters before training
		self.lr_end = learning_rate_end
		self.sigma_end = sigma_end
		self.lr_decrease = lr_decrease
		self.radius_decrease = radius_decrease
		
		if radius_decrease ==  "linear":
			self.sigma_time_const = ( (learning_rate - learning_rate_end))/max_epochs
		elif radius_decrease == "exp":
			self.sigma_time_const = (-1.)*max_epochs  /(np.log(learning_rate_end) - np.log(learning_rate))
			
		if lr_decrease ==  "linear":
			self.time_const = ((sigma - sigma_end))/max_epochs
		elif lr_decrease == "exp":
			self.time_const = (-1.)*max_epochs  /(np.log(sigma_end) - np.log(sigma))
		
		self.max_epochs = max_epochs
		self.sigma = sigma
		self.learning_rate = learning_rate
		self.initail_learning_rate = learning_rate
		self.initial_sigma = sigma
		self.batch_size = batch_size
		
		if C_INIT_SUCESS:
			self._train_c(prnt)
		else:
			self._train(prnt)
	
	def _update_weights_async(self,lr, elements, W):
		# decide which function is the faster one
		f = winning_neuron_asyinc if self.batch_size < self.outdim[0] * self.outdim[1] else winning_neuron_asyinc_large_batch
		
		pool_sze = self.pool_size
		#print(len(shaerd_weights))
		with Pool(processes=pool_sze) as pool:
			rad = 5*self.sigma
			iterations = [(elements[i*len(elements)//pool_sze:(i+1) * len(elements)//pool_sze],self.learning_rate,self.sigma,rad,self.periodic,self.outdim,self.Grid,W) for i in range(pool_sze)]
			new_maps = pool.map(f,iterations)
		
		for new_map in new_maps[1:]:
			new_maps[0][0] += new_map[0]
			new_maps[0][1] += new_map[1]	
		
		new_maps[0][0][(new_maps[0][1]==0).flatten()] *= 0
		new_maps[0][1][new_maps[0][1]==0] = 1
		return new_maps[0][0]/new_maps[0][1]


	def _update_weights(self,lr, elements, W):
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
			
			
			# Topological Neighbourhood Function
			h = lr * np.exp(-(self.d)/(2*self.sigma**2))[:, np.newaxis]  
			W += h*(x - W)
		return W
		
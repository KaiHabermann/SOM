#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>

double norm(double*mat1,double* vec, int x, int y, int dimx, int dimy, int input_dim){
	// computes the absolute distance for a vector to a vector inside a matrix (mat1 is the matrix represented as simple array of doubles)
	int i = x*dimy*input_dim + y * input_dim;
	double sum = 0; 
	int j = 0;
	int grenze = i + input_dim;
	for (;i< grenze;i++){
		sum += (mat1[i] - vec[j]) * (mat1[i] - vec[j]);
		j++;
	}
	return sum;
}


	

double* train_from_c(int dimx, int dimy, int input_dim,double** input_values,double* initial_weights,int input_size, double learning_rate, double sigma, double learning_rate_end, double sigma_end,int linear_rad, int linear_lr, int batchsize, int epochs,int prnt){
	double *weights;
	double sigma_dec, lr_dec, start_lr, start_sigma, gauss;
	
	double *temporay_weights = (double*) malloc(dimx*dimy*input_dim*sizeof(double));
	double *temporary_divisors = (double*) malloc(dimx*dimy*sizeof(double));
	double *vec;
	double nrm;
	double value;
	
	int max_vec_ind;
	
	int radius;
	int radx;
	int rady;
	start_sigma = sigma;
	start_lr = learning_rate;
	int x_min = 0;
	int y_min = 0;
	int ind = 0;
	
	// loadingbar setup
	char * loading_bar = malloc(40);
	memset(loading_bar,61,39);
	loading_bar[39] = 0;
	time_t start_t, end_t;
	double diff_t;

	
	weights = (double*) malloc(dimx*dimy*input_dim*sizeof(double));
	if (initial_weights == NULL){
		weights = (double*) malloc(dimx*dimy*input_dim*sizeof(double));
		for (int i = 0; i < dimx; i++) {
			for (int j = 0; j < dimy; j++){
				ind = rand()%input_size;
				//printf("%i, %i\n",ind,input_dim);
				for (int k = 0; k < input_dim; k++){
					
					weights[i*dimy* input_dim + j * input_dim + k] = input_values[ind][k];
				} 
			}
		}
	}
	else{
		weights = (double*) malloc(dimx*dimy*input_dim*sizeof(double));
		for (int i = 0; i < dimx; i++) {
			for (int j = 0; j < dimy; j++){
				//printf("%i, %i\n",ind,input_dim);
				for (int k = 0; k < input_dim; k++){
					
					weights[i*dimy* input_dim + j * input_dim + k] = initial_weights[i*dimy* input_dim + j * input_dim + k];
				} 
			}
		}
	}



	
	// for (int i = 0; i < input_size;i++)printf("(%lf, %lf, %lf)\n",input_values[i][0],input_values[i][1],input_values[i][2]);
	
	if (linear_rad) sigma_dec = ((double) (sigma - sigma_end))/epochs;
	else sigma_dec = (-1.)*epochs  /(log(sigma_end) - log(sigma));
	
	
	if (linear_lr) lr_dec = ((double) (learning_rate - learning_rate_end))/epochs;
	else lr_dec = (-1.)*epochs  /(log(learning_rate_end) - log(learning_rate));

	time(&start_t);

	for (int epoch = 0; epoch < epochs; epoch++){
		// epoch loop
		
		radius = (int) sqrt(fabs(2*sigma*sigma * log(1e-10)));
		
		for (int i = 0; i < dimx*dimy*input_dim; i++) temporay_weights[i] = 0;
		for (int i = 0; i < dimx*dimy; i++) temporary_divisors[i] = 0;
		
		max_vec_ind = (input_size < (epoch * batchsize )%input_size+ batchsize) ? input_size: ((epoch * batchsize)%input_size + batchsize);
		for (int vec_ind = (epoch*batchsize)%input_size; vec_ind < max_vec_ind; vec_ind++){
			// batch loop
			x_min = 0;
			y_min = 0;
			
			vec = input_values[vec_ind];
			value = norm(weights,vec,  0,  0,  dimx,  dimy,  input_dim);//running value for minimum
			// find minimum
			for(int x = 0; x < dimx; x++){
				for(int y = 0; y < dimy; y++){
					nrm = norm(weights,vec,  x,  y,  dimx,  dimy,  input_dim);
					if (nrm < value){
						value = nrm;
						y_min = y;
						x_min = x;
					}
				}
			}
			
			
			for(int x = 0; x < dimx; x++){
				for(int y = 0; y < dimy; y++){
					
					nrm = (x-x_min)*(x-x_min)+(y-y_min)*(y-y_min);
					if (nrm > radius*radius) continue;
					gauss = exp(-nrm/(2.*sigma*sigma));
					for (int k = 0; k < input_dim; k++){
						temporay_weights[x*dimy*input_dim+y*input_dim+k] += vec[k] *gauss ;
					}
					temporary_divisors[x*dimy+y] += gauss;
				}
			}
		

			
			
			
		}// batch loop end
		for (int i = 0; i < dimx; i++) {
			for (int j = 0; j < dimy; j++){
				if (temporary_divisors[i*dimy + j] != 0){
					for (int k = 0; k < input_dim; k++){
						
						weights[i*dimy* input_dim + j * input_dim + k] = (weights[i*dimy* input_dim + j * input_dim + k] + learning_rate *temporay_weights[i*dimy* input_dim + j * input_dim + k] / temporary_divisors[i*dimy + j] ) / (1.+learning_rate);
					} 
				}
				
			}
		}
	
		learning_rate = linear_lr ? start_lr - epoch*lr_dec : start_lr * exp((-1)*epoch/lr_dec);
		sigma = linear_rad ? start_sigma - epoch*sigma_dec : start_sigma * exp((-1)*epoch/sigma_dec);
		
		if (prnt) {		
			time(&end_t);	
			diff_t = difftime(end_t, start_t);
			diff_t /= epoch;
			diff_t *= (epochs - epoch - 1);
			loading_bar[(epoch * 39)/epochs] = '#';
			printf("%s ETA: %.2fs EPOCH: %d LR: %lf, SIGMA:%lf \r",loading_bar,diff_t,epoch,learning_rate,sigma);
		}

	}// epoch loop end
	if (prnt) printf("\n%s Time: %.2fs LR: %lf, SIGMA:%lf \n",loading_bar,difftime(end_t, start_t),learning_rate,sigma);
	
	return weights;
	
}


double* train_from_c_periodic(int dimx, int dimy, int input_dim,double** input_values,double* initial_weights, int input_size, double learning_rate, double sigma, double learning_rate_end, double sigma_end,int linear_rad, int linear_lr, int batchsize, int epochs,int prnt){
	double *weights;
	double sigma_dec, lr_dec, start_lr, start_sigma, gauss;
	
	
	
	double *temporay_weights = (double*) malloc(dimx*dimy*input_dim*sizeof(double));
	double *temporary_divisors = (double*) malloc(dimx*dimy*sizeof(double));
	double * vec;
	double nrm;
	double value;
	
	int max_vec_ind;
	
	int x_diff, y_diff;
	
	int radius;
	int radx;
	int rady;
	
	start_sigma = sigma;
	start_lr = learning_rate;

	int x_min = 0;
	int y_min = 0;
	int ind = 0;
	
	// setup loadingbar
	char * loading_bar = malloc(40);
	memset(loading_bar,61,39);
	loading_bar[39] = 0;
	time_t start_t, end_t;
	double diff_t;
	
	weights = (double*) malloc(dimx*dimy*input_dim*sizeof(double));
		if (initial_weights == NULL){
			weights = (double*) malloc(dimx*dimy*input_dim*sizeof(double));
			for (int i = 0; i < dimx; i++) {
				for (int j = 0; j < dimy; j++){
					ind = rand()%input_size;
					//printf("%i, %i\n",ind,input_dim);
					for (int k = 0; k < input_dim; k++){
						
						weights[i*dimy* input_dim + j * input_dim + k] = input_values[ind][k];
					} 
				}
			}
		}
		else{
			weights = (double*) malloc(dimx*dimy*input_dim*sizeof(double));
			for (int i = 0; i < dimx; i++) {
				for (int j = 0; j < dimy; j++){
					//printf("%i, %i\n",ind,input_dim);
					for (int k = 0; k < input_dim; k++){
						
						weights[i*dimy* input_dim + j * input_dim + k] = initial_weights[i*dimy* input_dim + j * input_dim + k];
					} 
				}
			}
		}





	
	// for (int i = 0; i < input_size;i++)printf("(%lf, %lf, %lf)\n",input_values[i][0],input_values[i][1],input_values[i][2]);
	
	if (linear_rad) sigma_dec = ((double) (sigma - sigma_end))/epochs;
	else sigma_dec = (-1.)*epochs  /(log(sigma_end) - log(sigma));
		
		
	if (linear_lr) lr_dec = ((double) (learning_rate - learning_rate_end))/epochs;
	else lr_dec = (-1.)*epochs  /(log(learning_rate_end) - log(learning_rate));
	time(&start_t);
	for (int epoch = 0; epoch < epochs; epoch++){
		// epoch loop
		
		
		radius = (int) sqrt(fabs(2*sigma*sigma * log(1e-10)));
		
		for (int i = 0; i < dimx*dimy*input_dim; i++) temporay_weights[i] = 0;
		for (int i = 0; i < dimx*dimy; i++) temporary_divisors[i] = 0;
		
		max_vec_ind = (input_size < (epoch * batchsize )%input_size+ batchsize) ? input_size: ((epoch * batchsize)%input_size + batchsize);
		for (int vec_ind = (epoch*batchsize)%input_size; vec_ind < max_vec_ind; vec_ind++){
			// batch loop
			x_min = 0;
			y_min = 0;
			
			vec = input_values[vec_ind];
			value = norm(weights,vec,  0,  0,  dimx,  dimy,  input_dim);//running value for minimum
			// find minimum
			for(int x = 0; x < dimx; x++){
				for(int y = 0; y < dimy; y++){
					nrm = norm(weights,vec,  x,  y,  dimx,  dimy,  input_dim);
					if (nrm < value){
						value = nrm;
						y_min = y;
						x_min = x;
					}
				}
			}
			
			
			for(int x = 0; x < dimx; x++){
				for(int y = 0; y < dimy; y++){
					x_diff = (x-x_min);
					y_diff = (y-y_min);
					y_diff = y_diff > dimy/2 ? abs(y_diff - dimy) : y_diff;
					x_diff = x_diff > dimx/2 ? abs(x_diff - dimx) : x_diff;

					nrm = x_diff*x_diff+y_diff*y_diff;
					if (nrm > radius*radius) continue;
					gauss = exp(-nrm/(2.*sigma*sigma));
					for (int k = 0; k < input_dim; k++){
						temporay_weights[x*dimy*input_dim+y*input_dim+k] += vec[k] *gauss ;
					}
					temporary_divisors[x*dimy+y] += gauss;
				}
			}
		

			
			
			
		}// batch loop end
		for (int i = 0; i < dimx; i++) {
			for (int j = 0; j < dimy; j++){
				if (temporary_divisors[i*dimy + j] != 0){
					for (int k = 0; k < input_dim; k++){
						
						weights[i*dimy* input_dim + j * input_dim + k] = (weights[i*dimy* input_dim + j * input_dim + k] + learning_rate *temporay_weights[i*dimy* input_dim + j * input_dim + k] / temporary_divisors[i*dimy + j] ) / (1.+learning_rate);
					} 
				}
				
			}
		}
	
		learning_rate = linear_lr ? start_lr - epoch*lr_dec : start_lr * exp((-1)*epoch/lr_dec);
		sigma = linear_rad ? start_sigma - epoch*sigma_dec : start_sigma * exp((-1)*epoch/sigma_dec);
		
		if (prnt) {		
			time(&end_t);	
			diff_t = difftime(end_t, start_t);
			diff_t /= epoch;
			diff_t *= (epochs - epoch - 1);
			loading_bar[(epoch * 39)/epochs] = '#';
			printf("%s ETA: %.2fs EPOCH: %d LR: %lf, SIGMA:%lf \r",loading_bar,diff_t,epoch,learning_rate,sigma);
		}
	}// epoch loop end
	if (prnt) printf("\n%s Time: %.2fs LR: %lf, SIGMA:%lf \n",loading_bar,difftime(end_t, start_t),learning_rate,sigma);
	
	return weights;
	
}



int* map_from_c(double * weights, double** input_values, int dimx, int dimy, int input_dim, int input_size){
	
	int* res = malloc(input_size * 2 * sizeof(int));
	double * vec;
	int x_min,y_min;
	double value, nrm;
	for (int vec_ind = 0; vec_ind < input_size; vec_ind++){
		x_min = 0;
		y_min = 0;
		
		vec = input_values[vec_ind];
		value = norm(weights,vec,  0,  0,  dimx,  dimy,  input_dim);//running value for minimum
		// find minimum
		for(int x = 0; x < dimx; x++){
			for(int y = 0; y < dimy; y++){
				nrm = norm(weights,vec,  x,  y,  dimx,  dimy,  input_dim);
				if (nrm < value){
					value = nrm;
					y_min = y;
					x_min = x;
				}
			}
		}
		res[2*vec_ind] = x_min;
		res[2*vec_ind +1 ] = y_min;
		

	}
	
	
	return res;
	
}


int* activation_from_c(double * weights, double** input_values, int dimx, int dimy, int input_dim, int input_size){
	
	int* res = malloc(dimx * dimy * sizeof(int));
	for (int i = 0; i < dimx * dimy;i++) res[i] = 0;
	
	double * vec;
	int x_min,y_min;
	double value, nrm;
	for (int vec_ind = 0; vec_ind < input_size; vec_ind++){
		x_min = 0;
		y_min = 0;
		
		vec = input_values[vec_ind];
		value = norm(weights,vec,  0,  0,  dimx,  dimy,  input_dim);//running value for minimum
		// find minimum
		for(int x = 0; x < dimx; x++){
			for(int y = 0; y < dimy; y++){
				nrm = norm(weights,vec,  x,  y,  dimx,  dimy,  input_dim);
				if (nrm < value){
					value = nrm;
					y_min = y;
					x_min = x;
				}
			}
		}
		res[x_min*dimy + y_min] += 1;
		
	}
	
	return res;
	
}


import sys
import numpy as np
import os,sys,inspect
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches as patches
import seaborn as sns
from datetime import datetime

# make sure SOM is in PATH
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0,current_dir)
from SOMs.bSOM import batch_SOM

def get_density(som,data):
    t0 = datetime.now()
    if isinstance(data,str):
        filepath = data
        data = pd.read_csv(filepath).values
        print("Loaded %s with shape %s"%(filepath,data.shape))
    hit_histogram = som.activation_matrix(data)
    density = hit_histogram/np.sum(hit_histogram) 
    dt = datetime.now()- t0
    print("Time taken %s s"%dt.seconds)
    return density

def toggle_style():
    plt.locator_params(axis='y', nbins=10)
    plt.locator_params(axis='x', nbins=10)
    plt.ylabel("y",rotation=0)
    plt.xlabel("x",rotation=0)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

def perform_density_fit(main_denstiy,MC_densities,dmain_density):
    from scipy.odr import Model, RealData, ODR
    def f(params,x):
        return sum(abs(p)*d for p,d in zip(params,MC_densities))
    
    linear = Model(f)
    data = RealData(np.zeros_like(main_denstiy), main_denstiy, sx=None, sy=dmain_density)
    regressor =  ODR(data, linear, beta0=[0 for _ in MC_densities])
    regressor.run()
    return regressor.output.beta

def plot_rel_density(wanted_index,MC_densities,weights,minimal_density = 1e-6):
    total_density = sum(w*d for w,d in zip(weights,MC_densities))
    mask = total_density != 0
    wanted_process = MC_densities[wanted_index] * weights[wanted_index]
    relative_density = np.zeros_like(wanted_process)
    relative_density[mask] = wanted_process[mask]/total_density[mask]
    # set all bins with too little data to 0
    relative_density[wanted_process < minimal_density] = 0
    sns.heatmap(relative_density)
    toggle_style()
    plt.show()
    return relative_density




def relative_density_test(data_path = "csv_files/2lep_complete.csv",
    model_path="csv_files/trainierte_soms/60x90.csv",
    MC_datasets = [ "csv_files/mc_files/HtoWW.csv",
                    "csv_files/mc_files/singletop.csv",
                    "csv_files/mc_files/singleW.csv",
                    "csv_files/mc_files/ttbar.csv",
                    "csv_files/mc_files/WW.csv",
                    "csv_files/mc_files/Ztautau.csv"]):
    """
    This tests the capability to get the relative density of a process on the SOM.
    It requires the user to provide all datasets for subprocesses.
    The example is done using ATLAS Open Data datasets.
    The datasets are not included in the package, but can be found in the ATLAS open data in the 2 lepton channel.    
    """
    map_dim = (60,90)
    try:
        values = np.loadtxt(data_path,delimiter=",",skiprows=1)
    except:
        raise(NotImplementedError("This test needs data installed at %s"%(data_path,)))
    decrease = "linear"
    # decrease = "exp"
    
    # som setup
    # PCA give the option to use PCA for initial Neuron distribution
    # pool_size only important, if train_async is used
    som = batch_SOM(map_dim,len(values[0]),values,
            PCA=True,periodic_boundarys=True,pool_size=8)
    
    # no training required, when we load an existing som
    try:
        if model_path is None:
            # train a new map, if wanted
            som.train(prnt = True,batch_size=500,learning_rate = 1.0,sigma_end=1.,learning_rate_end = 0.01,sigma=20,radius_decrease = decrease, lr_decrease = decrease,max_epochs=2000)
        else:
            som.load(model_path)
    except Exception as e:
        if model_path is not None:
            raise(NotImplementedError("This tests needs a pre trained SOM or model_path needs to be set to None to perform training"))
        else:
            raise(e)

    sns.heatmap(som.get_umatrix())
    plt.title("U-Matrix")
    toggle_style()
    plt.show()

    hit_histogram = som.activation_matrix(values)
    density = hit_histogram/np.sum(hit_histogram)
    ddensity = hit_histogram**0.5/np.sum(hit_histogram)
    sns.heatmap(density)
    plt.title("Data Density")
    toggle_style()
    plt.show()

    if MC_datasets is None or len(MC_datasets) == 0:
        # if no MC fit cant run
        exit(0)
    
    ddensity[ddensity==0] = 1/np.sum(hit_histogram)
    MC_densities = [get_density(som,data) for data in MC_datasets]
    
    # if you got the actual weights for each process, you can just use them here
    weights = perform_density_fit(density,MC_densities,ddensity)

    # this can now be used to isolate a dataset
    relative_density = plot_rel_density(0,MC_densities,weights)
    
    for name, weight in zip(MC_datasets,weights):
        print("%s with weight %.2e"%(name,weight))
    print("Sum: %.2f"%sum(weights))


if __name__=="__main__":
    #relative_density_test()
    relative_density_test(model_path=None)
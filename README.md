# BKMR-MHMC

## Setting up the environment
1. [Install miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)(if not already installed)
2. Set up the virtual environment<br>        

conda env create -f environment.yml<br> 
conda activate bkmr_mhmc<br>



if jax and jaxlib not successfully installed, [install it](https://github.com/google/jax#installation) here<br>

pip install jax==0.3.0<br>
pip install jaxlib==0.3.0<br>

if the installation of jaxlib still not successful, put the jaxlib folder under the your_environment_location/Lib/site-packages folder and install the dependent modules:

pip install joblib <br>
pip install numpyro <br>

## Reproducing the simulation results
Use the scripts in the scripts folder to reproduce results in the paper.




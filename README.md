# BKMR-MHMC

## Setting up the environment
1. [Install miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)(if not already installed)
2. Set up the virtual environment//
conda env create -f environment.yml
conda activate bkmr_mhmc

if jax and jaxlib not successfully installed, [install it](https://github.com/google/jax#installation) here

pip install jax==0.3.0
pip install jaxlib==0.3.0

if the installation of jaxlib still not successful, put the jaxlib folder under the your_environment_location/Lib/site-packages folder

pip install joblib
pip install numpyro



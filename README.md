# BKMR-MHMC

## Setting up the environment
1. [Install miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)(if not already installed)
2. Set up the virtual environment<br>        
<html>
<body>
<p>conda env create -f environment.yml    <br> 
conda activate bkmr_mhmc<br>  </p>
</body>
</html>



          

if jax and jaxlib not successfully installed, [install it](https://github.com/google/jax#installation) here<br>

pip install jax==0.3.0<br>
pip install jaxlib==0.3.0<br>

if the installation of jaxlib still not successful, put the jaxlib folder under the your_environment_location/Lib/site-packages folder

pip install joblib
pip install numpyro



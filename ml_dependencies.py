# checking if various packages for ML were installed properly in the conda environment

# use conda install -c anaconda scipy
import scipy
print('scipy: %s' % scipy.__version__)
# above command should take care of numpy too
import numpy
print('numpy: %s' % numpy.__version__)
# use conda install -c conda-forge matplotlib 
import matplotlib
print('matplotlib: %s' % matplotlib.__version__)
# conda install pandas
import pandas
print('pandas: %s' % pandas.__version__)
# conda install -c anaconda statsmodels
import statsmodels
print('statsmodels: %s' % statsmodels.__version__)
# conda install -c anaconda scikit-learn
import sklearn
print('sklearn: %s' % sklearn.__version__)

# conda install -c conda-forge librosa
import librosa
print('librosa: %s' % librosa.__version__)
# pip install audiomentations
import audiomentations
print('audiomentations: %s' % audiomentations.__version__)

# conda install -c anaconda tensorflow-gpu
import tensorflow
print('tensorflow: %s' % tensorflow.__version__)
# conda install -c anaconda keras-gpu
import keras
print('keras: %s' % keras.__version__)
# pip install wandb
import wandb
print('wandb: %s' % wandb.__version__)
# see what devices tensorflow recognizes (cpu/gpu)
print(tensorflow.config.list_physical_devices())
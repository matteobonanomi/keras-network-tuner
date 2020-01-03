from tensorflow.keras.datasets import boston_housing
import logging
import yaml
from source.GridSearchKeras import *
from source.utils.auxiliary_functions import *
from sklearn.preprocessing import StandardScaler
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# SET LOGGER
logging._warn_preinit_stderr = 0
logger = logging.getLogger()
# create file handler that logs debug and higher level messages
_file_path = '../logs/'
_file_name = pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M_')+'notebook.log'
fh = logging.FileHandler(_file_path + _file_name)
fh.setLevel(logging.INFO)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
# add the handlers to logger
logger.addHandler(ch)
logger.addHandler(fh)
logger.setLevel(logging.INFO)

# CONFIG FILE
with open("../config/keras_config.yaml", 'r') as stream:
    kerasgrid_config = yaml.safe_load(stream)

# PREPROCESSING
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()
X_scaler = StandardScaler()
X_scaler.fit(X_train)
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)

# TRAIN
gridKeras = KerasTuner(kerasgrid_config)
gridKeras.set_callbacks(generate_callbacks)
gridKeras.initialize_network(create_model)
gridKeras.initialize_grid()
gridKeras.fit(X_train, y_train, X_test, y_test)

# TEST
gridKeras.predict(X_test, y_test)

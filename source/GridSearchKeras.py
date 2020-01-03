import numpy as np
import pandas as pd
import os
import logging
import yaml
from source.utils.auxiliary_functions import *
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.regularizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from source.utils.auxiliary_functions import *


logger = logging.getLogger()


class KerasTuner(object):
    """
    Grid Search CV class for Keras-based neural network models
    """

    def __init__(self, config=None):
        """
        Initialize class attri
        :param config:
        """
        self._generate_callbacks = return_empty_list
        self._grid_params = None
        self._model = None
        self._cv_params = None
        self._grid = None
        self._grid_results = None
        self._is_model_initialized = False
        self._is_grid_initialized = False
        self._is_trained = False
        self._config = None

        if config is None:
            logger.info('No default config dictionaty was loaded')
        else:
            self.set_config_params(config)
            logger.info('Default config dictionaty was initialized')

        return

    def set_config_params(self, config=None):
        """
        Initialize configuration parameters
        Follow sample keras_config.yaml to fill network architecture hyper parameters
        :param config:
        :return:
        """

        if config is None:
            logger.info('Default config parameters will be used')
        else:
            config['grid_params']['network'] = convert_yaml2grid(config['grid_params']['network'])
            self.set_grid_params(config['grid_params'])
            self.set_cv_params(config['cv_params'])
            self._config = config
            logger.info('Passed config parameters will be used')

        return

    def set_grid_params(self, params):
        """
        Set grid search params to tune neural network hyperparameters:
        1) Network architecture
        2) Epochs
        3) Batch size
        4) Optimizer

        :param params: dictionary (see keras_config.yaml for details)
        :return:
        """

        self._grid_params = params

        return

    def set_callbacks(self, callbacks_function):
        """
        Set callback function.
        This function must provide a list of Keras callbacks to be applied when fitting the model
        Default function returns an empty list (no callbacks applied)
        :param callbacks_function: function object
        :return:
        """
        self._generate_callbacks = callbacks_function

        return

    def set_cv_params(self, params):
        """
        Set scikit-learn cross validation paramters:
        1) n_jobs
        2) cv
        3) verbose
        4) scoring
        :param params: dictionary (see keras_config.yaml for details)
        :return:
        """
        self._cv_params = params

        return

    def initialize_network(self, network_function):
        """
        Initialize KerasRegressor object that makes Keras model compatible with scikit-learn library
        :param network_function: any function that returns a compiled Keras model, taking hyper parameters as input
        :return:
        """

        self._model = KerasRegressor(build_fn=network_function)
        self._is_model_initialized = True

        return

    def initialize_grid(self):
        """
        Initialize grid search object, given model function, grid search params and cross validation params
        Two tuning strategies available:
        1) Grid Search CV --> cross validation + full grid search
        2) Randomized Search CV --> cross validation + search on a random combination of hyper parameters
        Use Randomized Search for fast model training, Grid Search for good model quality
        :return:
        """

        assert self._is_model_initialized, 'Keras model was not instantiated (initialize_network method)'
        assert self._config is not None, 'No config yaml file have been initialized'
        assert self._cv_params is not None, 'No cross validation params have been initialized'
        _cv = self._cv_params['cv']
        _verb = self._cv_params['verb']
        _scoring = self._cv_params['scoring']
        _n_jobs = self._cv_params['n_jobs']
        _mode = self._config['mode']
        if _mode == 'grid_search':
            logger.info('Grid Search mode is selected')
            self._grid = GridSearchCV(estimator=self._model,
                                      param_grid=self._grid_params,
                                      scoring=_scoring,
                                      cv=_cv,
                                      verbose=_verb,
                                      n_jobs=_n_jobs)
        elif _mode == 'randomized_search':
            _n_iter = self._config['cv_params']['n_iter']
            logger.info('Randomized Search mode with {} iterations is selected'.format(_n_iter))
            self._grid = RandomizedSearchCV(self._model,
                                            self._grid_params,
                                            n_iter=_n_iter,
                                            scoring=_scoring,
                                            cv=_cv,
                                            verbose=_verb,
                                            n_jobs=_n_jobs)
        else:
            logger.error('Invalid KerasTuner mode: {}'.format(_mode))
            raise
        self._is_grid_initialized = True
        logger.info('KerasTuner object is instantiated')

        return

    def fit(self, X_train, y_train, X_test, y_test):

        assert self._is_grid_initialized, 'Grid Search object was not instantiated (initialize_grid method)'

        self._grid_params['input_layer_shape'] = [X_train.shape[1]]

        self._grid_results = self._grid.fit(X_train, y_train,
                                            validation_data=(X_test, y_test),
                                            verbose=0,
                                            callbacks=self._generate_callbacks())

        print("Best: %f using %s" % (self._grid_results.best_score_, self._grid_results.best_params_))

        return

    def predict(self, X_test, y_test):

        y_pred = self._grid.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        logger.info('MAE on test: {:.2f}, RMSE on test: {:.2f}'.format(mae, rmse))

        return

    def load(self):

        return

    def dump(self):

        return


def create_model(input_layer_shape=None,
                 network=((10, 'relu', 0.1), (1, 'linear', 0)),
                 optimizer='adam'):
    n = len(network)
    model = Sequential()
    model.add(Dropout(network[0][2]))
    model.add(Dense(network[0][0], input_dim=input_layer_shape, activation=network[0][1]))
    if n > 1:
        for _layer in range(1, n):
            model.add(Dropout(network[_layer][2]))
            model.add(Dense(network[_layer][0],
                            activation=network[_layer][1],
                            kernel_regularizer=l1_l2(network[_layer][3], network[_layer][4])))

    model.compile(loss='mae', optimizer=optimizer, metrics=['accuracy'])
    return model


def generate_callbacks():
    early = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=10,
                          verbose=0, mode='auto', baseline=None, restore_best_weights=False)
    reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10,
                                 verbose=0, mode='auto', min_delta=0.001, cooldown=0, min_lr=0)

    return [early, reducelr]

import numpy as np
import pandas as pd
import os
import logging
import yaml

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.regularizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from source.utils.auxiliary_functions import *


logger = logging.getLogger()


class gridSearchKerasNN(object):

    def __init__(self, config=None):

        if config is None:
            logger.info('No default config dictionaty was loaded')
        else:
            self.set_config_params(config)
            self._config = config
            logger.info('Default config dictionaty was initialized')

        self._generate_callbacks = return_empty_list

        return

    def set_grid_params(self, params):

        self._grid_params = params

        return

    def set_callbacks(self, callbacks_function):

        self._generate_callbacks = callbacks_function

        return

    def set_cv_params(self, params):

        self._cv_params = params

        return

    def set_config_params(self, config=None):

        if config is None:
            logger.info('Default config parameters will be used')
        else:
            self.set_grid_params(config['grid_params'])
            self.set_cv_params(config['cv_params'])
            logger.info('Passed config parameters will be used')

        return

    def initialize_network(self, network_function):

        self._model = KerasRegressor(build_fn=network_function)

        return

    def initialize_grid(self):

        _cv = self._cv_params['cv']
        _verb = self._cv_params['verb']
        _scoring = self._cv_params['scoring']
        _n_jobs = self._cv_params['n_jobs']

        self._grid = GridSearchCV(estimator=self._model,
                                  param_grid=self._grid_params,
                                  scoring=_scoring,
                                  cv=_cv,
                                  verbose=_verb,
                                  n_jobs=_n_jobs)
        logger.info('GridSearchCV object is instantiated')

        return

    def fit(self, X_train, y_train, X_test, y_test):

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

from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
import numpy as np


class Regressor(ABC):
    """ A generic regressor class used in forecasting"""

    def __init__(self):
        """ initialize regressor"""
        pass

    @abstractmethod
    def fit(self, X, Y):
        """
        train the regressor
        :param X: set of sample inputs
        :type X: np.ndarray
        :param Y: set of sample outputs
        :type Y: np.ndarray
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        predict for a sample of inputs
        :param X: a set of inputs
        :type X: np.ndarray
        :return: a set out next step outputs
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def forecast(self, curve):
        """
        forecast the entire future for one input curve
        :param curve: time series of brightness in one passband
        :type curve: np.ndarray
        :return: the future multi-step forecast
        :rtye: np.ndarray
        """
        pass


class LinearRegressor(Regressor):
    """ A simple linear regressor"""

    def __init__(self, historical_window=5):
        """
        initialize regresor
        :param historical_window: how many historical steps to use in predicting the future
        :type historical_window: int
        """
        Regressor.__init__(self)
        self.model = None
        self.goes_k = historical_window

    def fit(self, X, Y):
        """
        train the regressor
        :param X: set of sample inputs
        :type X: np.ndarray
        :param Y: set of sample outputs
        :type Y: np.ndarray
        """
        self.model = LinearRegression().fit(X, Y)

    def predict(self, X):
        """
        predict for a sample of inputs
        :param X: a set of inputs
        :type X: np.ndarray
        :return: a set out next step outputs
        :rtype: np.ndarray
        """
        return self.model.predict(X)

    def forecast(self, curve):
        """
        forecast the entire future for one input curve
        :param curve: time series of brightness in one passband
        :type curve: np.ndarray
        :return: the future multi-step forecast
        :rtye: np.ndarray
        """
        forecast = []
        history = curve[:self.goes_k]
        for i in range(self.goes_k, curve.shape[0]):
            forecast.append(self.model.predict(history.reshape((1, -1))))
            history = np.append(history[1:], forecast[-1])
        return np.array(forecast)


from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression as skLinearRegression
from sklearn.ensemble import RandomForestRegressor as skRandomForestRegressor
import numpy as np
import enum


class RegressorMode(enum.Enum):
    """
    The mode the regressor is running in:
        Recursive predicts just the next step and moves forward until a sufficient future window is predicted
        Many models trains a different model for each lag
        Many outputs trains one model that outputs all the time lags
    """
    RecursiveModel = 0
    ManyModels = 1
    ManyOutputs = 2


class Regressor(ABC):
    """ A generic regressor class used in forecasting"""

    def __init__(self, mode, historic_horizon=15, future_horizon=15):
        """ initialize regressor"""
        self.models = []
        self.mode = mode
        self.historic_horizon = historic_horizon
        self.future_horizon = future_horizon

    def fit(self, X, y):
        """
        train the regressor
        :param X: set of sample inputs
        :type X: np.ndarray
        :param y: set of sample outputs
        :type y: np.ndarray
        """
        if self.mode == RegressorMode.RecursiveModel:
            return self._fit_recursive(X, y)
        elif self.mode == RegressorMode.ManyModels:
            return self._fit_many_models(X, y)
        elif self.mode == RegressorMode.ManyOutputs:
            return self._fit_many_outputs(X, y)
        else:
            raise NotImplemented("You requested a mode that was not implemented. See RegressorMode enumeration.")

    @abstractmethod
    def _fit_recursive(self, X, y):
        pass

    @abstractmethod
    def _fit_many_models(self, X, y):
        pass

    @abstractmethod
    def _fit_many_outputs(self, X, y):
        pass

    def predict(self, X):
        """
        predict for a sample of inputs
        :param X: a set of inputs
        :type X: np.ndarray
        :return: a set out next step outputs
        :rtype: np.ndarray
        """
        if self.mode == RegressorMode.RecursiveModel:
            return self._predict_recursive(X)
        elif self.mode == RegressorMode.ManyModels:
            return self._predict_many_models(X)
        elif self.mode == RegressorMode.ManyOutputs:
            return self._predict_many_outputs(X)
        else:
            raise NotImplemented("You requested a mode that was not implemented. See RegressorMode enumeration.")

    def _predict_recursive(self, X):
        """
        Predicts using the recursive mode
        :param X: either a single feature set or a database
        :return: the prediction out to the window
        """
        if len(X.shape) == 2:  # data base
            return np.stack([self._predict_recursive(x) for x in X])
        else:  # single feature set
            history = np.zeros(self.historic_horizon + self.future_horizon)
            history[:self.historic_horizon] = X
            for i in range(self.historic_horizon, self.historic_horizon + self.future_horizon):
                history[i] = self.models[0].predict(history[i - self.future_horizon:i].reshape(1, -1))
            return history[-self.future_horizon:]

    def _predict_many_models(self, X):
        """
        Predicts using the many models approach
        :param X: either a single feature set or a database
        :return: the prediction out to the window
        """
        if len(X.shape) == 2:
            return np.stack([self._predict_many_models(x) for x in X])
        else:  # a single sample was passed
            return np.concatenate([model.predict(X.reshape(1, -1)) for model in self.models])

    def _predict_many_outputs(self, X):
        """
        Predicts using the many outputs approach
        :param X: either a single feature set or a database of samples
        :return: the prediction out to the window
        """
        return self.models[0].predict(X)

    @classmethod
    def rmse(cls, true_values, predictions):
        """
        Root mean squared
        :param true_values: true light curve
        :param predictions: predicted light curve
        :return: root mean squared error
        """
        return np.sqrt(mean_squared_error(true_values, predictions))


class BaselineRegressor(Regressor):
    class Persistence:
        def __init__(self):
            pass

        # make a persistence forecast
        def predict(self, X):
            if len(X.shape) == 1:
                return X[-1]
            else:
                return np.stack([self.predict(x) for x in X])

    """
    A simple persistence forecaster
    """
    def __init__(self, historic_horizon=15, future_horizon=15):
        """
        initialize regressor
        :param historic_horizon: how many historical steps to use in predicting the future
        :type historic_horizon: int
        :param future_horizon: how many future steps to predict
        :type future_horizon: int
        """
        Regressor.__init__(self, RegressorMode.RecursiveModel,
                           historic_horizon=historic_horizon, future_horizon=future_horizon)
        self.models.append(self.Persistence())

    def _fit_recursive(self, X, y):
        pass

    def _fit_many_models(self, X, y):
        pass

    def _fit_many_outputs(self, X, y):
        pass


class LinearRegressor(Regressor):
    """ A simple linear regressor"""

    def __init__(self, mode, historic_horizon=15, future_horizon=15):
        """
        initialize regressor
        :param historic_horizon: how many historical steps to use in predicting the future
        :type historic_horizon: int
        :param future_horizon: how many future steps to predict
        :type future_horizon: int
        """
        Regressor.__init__(self, mode, historic_horizon=historic_horizon, future_horizon=future_horizon)

    def _fit_recursive(self, X, y):
        self.models.append(skLinearRegression().fit(X, y[:, 0]))

    def _fit_many_models(self, X, y):
        for i in range(self.future_horizon):
            self.models.append(skLinearRegression().fit(X, y[:, i]))

    def _fit_many_outputs(self, X, y):
        self.models.append(skLinearRegression().fit(X, y[:, :self.future_horizon]))


class RandomForestRegressor(Regressor):
    def __init__(self, mode, historic_horizon=15, future_horizon=15, n_estimators=100, criterion="mse",
                 max_depth=None, min_samples_split=2, min_samples_leaf=1):
        """

        :param mode:
        :param historic_horizon:
        :param future_horizon:
        """
        Regressor.__init__(self, mode, historic_horizon=historic_horizon, future_horizon=future_horizon)

        # save the forest parameters
        self.forest_params = {'n_estimators': n_estimators,
                              'criterion': criterion,
                              'max_depth': max_depth,
                              'min_samples_split': min_samples_split,
                              'min_samples_leaf': min_samples_leaf}

    def _fit_recursive(self, X, y):
        self.models.append(skRandomForestRegressor(**self.forest_params).fit(X, y[:, 0]))

    def _fit_many_models(self, X, y):
        for i in range(self.future_horizon):
            self.models.append(skRandomForestRegressor(**self.forest_params).fit(X, y[:, i]))

    def _fit_many_outputs(self, X, y):
        self.models.append(skRandomForestRegressor(**self.forest_params).fit(X, y[:, :self.future_horizon]))

import inspect

import numpy as np
from baseconvert import base
from scipy.stats import special_ortho_group
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import check_is_fitted

from .solver import GeneralizedElasticNetSolver


class GeneralizedElasticNetRegressor(BaseEstimator, RegressorMixin):
    """
    Generalized elastic net regressor
    """

    def __init__(self, beta, lam_1=0.0, lam_2=0.0, lowbo=None, upbo=None, ds=None, sigma_ds=None, wvec=None,
                 random_state=None,
                 sigma_choice=0, sigma_choice_base=None, sigma_choice_up=10 ** 5,
                 w_choice=0, w_choice_base=None, w_choice_up=10 ** 5,
                 err_tol=1e-8, verbose=False, text_fr=200):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        self.p = len(beta)

        for arg, val in values.items():
            setattr(self, arg, val)

        if self.sigma_choice_base is None:
            base = 2
            while base ** self.p <= self.sigma_choice_up:
                base += 1
            self.sigma_choice_base = base - 1

        if self.w_choice_base is None:
            base = 2
            while base ** self.p <= self.w_choice_up:
                base += 1
            self.w_choice_base = base - 1

    def _generate_combination(self, p, k, choices=[0, 1]):
        assert (k < 2 ** p), 'k cannot be bigger than 2**p'
        res = np.empty(p)
        for i in range(p):
            den = 2 ** (i + 1)
            if k >= den:
                num = k - den
            else:
                num = k
            temp = np.floor(num / 2 ** i)
            res[i] = choices[int(np.mod(temp, 2))]
        return res

    @staticmethod
    def decimal2combination(decimal, p, bases):
        # decimal<bases**p
        if decimal >= bases ** p:
            raise ValueError('decimal should less than bases**p')
        comb = base(int(decimal), 10, bases)
        comb = max((p - len(comb)), 0) * [0] + [item for item in comb]
        return np.array(comb).astype('float')

    @staticmethod
    def combination2decimal(comb, bases):
        dec = base("".join(comb.astype(str)), bases, 10, string=True)
        return int(dec)

    def fit(self, X, y=None):
        n, p = X.shape

        solver = GeneralizedElasticNetSolver()

        if self.random_state is None:
            self.random_state = 10

        if self.sigma_ds is None:
            if self.ds is None:
                # self.ds=np.ones(p, dtype=np.float)
                ds = GeneralizedElasticNetRegressor.decimal2combination(self.sigma_choice, p, self.sigma_choice_base)
                if sum(ds) != 0:
                    ds = ds / sum(ds)
                ortho_mat = special_ortho_group.rvs(dim=p, random_state=self.random_state)
                self.sigma_mat = ortho_mat @ np.diag(ds) @ ortho_mat.T
            else:
                assert (self.ds.shape[0] == p), 'Please make sure the dimension of the dataset matches!'
                ortho_mat = special_ortho_group.rvs(dim=p, random_state=self.random_state)
                self.sigma_mat = ortho_mat @ np.diag(self.ds) @ ortho_mat.T
        else:
            self.sigma_mat = np.diag(self.sigma_ds)

        if self.wvec is None:
            # self.wvec=np.ones(p, dtype=np.float)/p
            self.newwvec = GeneralizedElasticNetRegressor.decimal2combination(self.w_choice, p, self.w_choice_base)
            if sum(self.newwvec) != 0:
                self.newwvec = self.newwvec / sum(self.newwvec)
        else:
            self.newwvec = self.wvec

        if self.lowbo is None:
            self.lowbo = np.repeat(0.000001, p)

        if self.upbo is None:
            self.upbo = np.repeat(float('inf'), p)

        self.coef_ = solver.solve(X, y, self.lam_1, self.lam_2, self.lowbo, self.upbo, self.newwvec, self.sigma_mat,
                                  self.err_tol, self.verbose, self.text_fr)

        return self

    def _decision_function(self, X):
        check_is_fitted(self, "coef_")
        return np.dot(X, self.coef_)

    def predict(self, X, y=None):

        return self._decision_function(X)

    def score(self, X, y=None, sample_weight=None):
        return mean_squared_error(X.dot(self.beta), self.predict(X), sample_weight=sample_weight)

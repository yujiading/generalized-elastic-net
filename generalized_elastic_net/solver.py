import copy

import numpy as np
from math import sqrt


class GeneralizedElasticNetSolver(object):
    """
    Generalized Elastic Net solver
    """

    def gmulup_solve(self, Amat, lvec, bvec, dvec, v0, err_tol=1e-8, text='Off', text_fr=200):
        """
        The generalized multiplcative updates solver

        :param Amat: A strictly positive definite matrx
        :type Amat: ndarray
        :param lvec: The upper bound vector
        :type lvec: array
        :param bvec: A vector
        :type bvec: array
        :param dvec: The regularization term
        :type dvec: array
        :param v0: The shifting vector
        :type v0: array
        :param err_tol: The tolerance vector
        :type err_tol: array
        :param text: If equals to 'On', print difference to the result of the last iteration, otherwise no print
        :type text: str
        :param text_fr: Text-printing frequency, only works when text is 'On'
        :type text_fr: int
        :return: Solution of loss function
        :rtype: array
        """

        A_plus = copy.deepcopy(Amat)
        A_plus[A_plus < 0] = 0

        A_minus = copy.deepcopy(Amat)
        A_minus[A_minus > 0] = 0
        A_minus = abs(A_minus)

        v = np.array([1.0 for x in range(len(bvec))])

        old_v = np.array([0 for x in range(len(bvec))])
        v0 = v0.astype(float)
        v0[np.where(v0 == 0)] = 0.00000001

        updateFactor = np.array([1.0 for x in range(len(bvec))])
        count = 0
        while (((old_v - v) ** 2).sum() > err_tol):
            v[np.where(v == 0)] = 0.00000001
            updateFactor0 = v0 / v
            updateFactor0[np.where(updateFactor0 == 0)] = 0.00000001
            dFa = np.array(A_plus.dot(v))
            dFb = copy.deepcopy(bvec)
            dFc = np.array(A_minus.dot(v))
            for i in range(len(bvec)):
                if dFa[i] == 0:
                    dFa[i] = 0.00000001
                if dFa[i] * updateFactor0[i] + (dFb[i] - dvec[i]) - dFc[i] / updateFactor0[i] > 0:
                    updateFactor[i] = float(
                        (-(dFb[i] - dvec[i]) + sqrt((dFb[i] - dvec[i]) ** 2 + 4 * dFa[i] * dFc[i])) / (2 * dFa[i]))
                elif dFa[i] * updateFactor0[i] + (dFb[i] + dvec[i]) - dFc[i] / updateFactor0[i] < 0:
                    updateFactor[i] = float(
                        (-(dFb[i] + dvec[i]) + sqrt((dFb[i] + dvec[i]) ** 2 + 4 * dFa[i] * dFc[i])) / (2 * dFa[i]))
                else:
                    updateFactor[i] = updateFactor0[i]
            if np.count_nonzero(~np.isnan(updateFactor)) == len(bvec):
                old_v = copy.deepcopy(v)
                v = np.minimum(lvec, updateFactor * v)
            else:
                break
            if (count % text_fr == 0) & (text == 'On'):
                print(((old_v - v) ** 2).sum())
            count += 1

        return v

    def solve(self, Xmat, Yvec, lam_1, lam_2, lowbo, upbo, wvec, Sigma, err_tol=1e-8, text='Off', text_fr=200):
        """
        Solve the linear regression by the generalized muliplcative updates
        :param Xmat: The design matrix
        :type Xmat: ndarray
        :param Yvec: The response vector
        :type Yvec: array
        :param lam_1: The l1 regularization term
        :type lam_1: float
        :param lam_2: The l2 regularization term
        :type lam_2: float
        :param lowbo: Lower bound vector
        :type lowbo: array
        :param upbo: Upper bound vector
        :type upbo: array
        :param wvec: Generalized l1 regularization vector
        :type wvec: array
        :param Sigma: Generalized l2 regularization matrix
        :type Sigma: ndarray
        :param err_tol: Tolerance
        :type err_tol: float
        :param text: If equals to 'On', print difference to the result of the last iteration, otherwise no print
        :type text: str
        :param text_fr: Text-printing frequency, only works when text is 'On'
        :type text_fr: int
        :return: Solution of the linear regression
        :rtype: array
        """

        p = Xmat.shape[1]
        Amat = Xmat.transpose().dot(Xmat) + lam_2 * Sigma
        bvec = 2 * Amat.dot(lowbo) - 2 * Xmat.transpose().dot(Yvec)
        Amat = 2 * Amat
        dvec = lam_1 * wvec
        v0 = np.maximum(0, -lowbo)
        lvec = upbo - lowbo
        v = self.gmulup_solve(Amat, lvec, bvec, dvec, v0, err_tol, text, text_fr)

        beta = v + lowbo
        return beta

# To solve prob: Yvec=Xmat*beta

# Example of input
# Xmat = np.random.randn(N,K)
# Yvec = np.random.randn(N)
# lam_1 = 0.0034
# lam_2 = 0
# Sigma = np.diag([1]*K)
# wvec = np.ones(K)
# lowbo = -1*np.ones(N)
# upbo = 1*np.ones(N)

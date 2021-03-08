# Generalized Elastic Net Library
- Version:
- Released: 
- Publication: [arXiv]

Generalized elastic net is a penalty method for variable selection and regularization in high-dimensional sparse linear models. It generalizes and outperforms the lasso, ridge, and nonnegative elastic net through (1) capturing the penalty weights for individual features or/and interactions between any two features; (2) controling the range of the coefficients.

The algorithm is available through this public Python library. It applys multiplicative updates on a quadratic programming problem but contains absolute values of variables and a rectangle-range constraint. The algorithm is shown to converge monotonically to the global in the publication.

## To install
The generalized elastic net library requires Python 3 and is pip friendly. To get started, simply do:
'''
pip install generalized-elastic-net
'''
or check out the code from out GitHub repository.
You can now use the package in Python with:
'''
import generalized-elastic-net
'''

## Examples

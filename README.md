# Portfolio Optimization with Cumulative Prospect Theory Utility via Convex Optimization

This repo accompanies our [paper](https://arxiv.org/abs/2209.03461).

## Installation

The `cptopt` package can be installed using `pip` as follows

```python
pip install git+https://github.com/cvxgrp/cptopt.git
```

## Minimum working example
We are unable to provide the full data set used in the paper for licensing reasons. We, therefore, give a minimum working example using simulated data below.
```python
import numpy as np
from scipy.stats import multivariate_normal as normal

from cptopt.optimizer import MinorizationMaximizationOptimizer, ConvexConcaveOptimizer, \
    MeanVarianceFrontierOptimizer, GradientOptimizer
from cptopt.utility import CPTUtility

# Generate returns
corr = np.array([
    [1, -.2, -.4],
    [-.2, 1, .5],
    [-.4, .5, 1]
])
sd = np.array([.01, .1, .2])
Sigma = np.diag(sd) @ corr @ np.diag(sd)

np.random.seed(0)
r = normal.rvs([.03, .1, .193], Sigma, size=100)

# Define utility function
utility = CPTUtility(
    gamma_pos=8.4, gamma_neg=11.4,
    delta_pos=.77, delta_neg=.79
)

initial_weights = np.array([1/3, 1/3, 1/3])

# Optimize
mv = MeanVarianceFrontierOptimizer(utility)
mv.optimize(r, verbose=True)

mm = MinorizationMaximizationOptimizer(utility)
mm.optimize(r, initial_weights=initial_weights, verbose=True)

cc = ConvexConcaveOptimizer(utility)
cc.optimize(r, initial_weights=initial_weights, verbose=True)

ga = GradientOptimizer(utility)
ga.optimize(r, initial_weights=initial_weights, verbose=True)
```
The optimal weights can then be accessed via the `weights` property.
```py
mv.weights
mm.weights
cc.weights
ga.weights
```

## Citing
If you want to reference our paper in your research, please consider citing us by using the following BibTeX:

```BibTeX
@misc{luxenberg2022cptopt,
  author = {Luxenberg, Eric and Schiele, Philipp and Boyd, Stephen},
  doi = {10.48550/ARXIV.2209.03461},
  title = {Portfolio Optimization with Cumulative Prospect Theory Utility via Convex Optimization},
  url = {https://arxiv.org/abs/2209.03461},
  year = {2022}
}
```


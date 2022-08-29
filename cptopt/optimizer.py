from __future__ import annotations

import abc
import copy
import time
from abc import abstractmethod
from typing import Tuple, Optional, Union

import cvxpy as cp
import numpy as np
import torch

from cptopt.utility import CPTUtility, CPTUtilityMM, CPTUtilityCC, CPTUtilityGA


class CPTOptimizer(abc.ABC):
    def __init__(self, utility: CPTUtility, max_iter: int = 1000):
        utility = copy.deepcopy(utility)
        self.utility = utility
        self.max_iter = max_iter
        self._weights = None
        self._weights_history = None
        self._wall_time = None

    @abstractmethod
    def optimize(self, r: np.array, verbose: bool = False) -> None:
        pass

    @property
    def weights(self) -> np.array:
        assert self._weights is not None
        return self._weights

    @property
    def weights_history(self) -> np.array:
        assert self._weights_history is not None
        return self._weights_history

    @property
    def wall_time(self) -> np.array:
        assert self._wall_time is not None
        return self._wall_time - np.min(self._wall_time)


class MinorizationMaximizationOptimizer(CPTOptimizer):
    def __init__(self, utility: Union[CPTUtility, CPTUtilityMM], max_iter: int = 1000):
        super().__init__(utility, max_iter)
        if isinstance(self.utility, CPTUtility):
            CPTUtilityMM.convert_to_class(self.utility)

    def optimize(
        self,
        r: np.array,
        verbose: bool = False,
        solver: Optional[str] = None,
        initial_weights: Optional[np.array] = None,
        eps: float = 1e-4,
        max_time: float = np.inf,
    ) -> None:

        w_prev = (
            initial_weights if initial_weights is not None else np.ones(r.shape[1]) / r.shape[1]
        )
        weight_history = [w_prev]
        wall_time = [time.time()]
        best_w = w_prev
        best_ob, _, _ = self.utility.evaluate(w_prev, r)

        if verbose:
            print("#" * 50)
            print(" Starting MMOptimizer ".center(50, "#"))
            print("#" * 50 + "\n\n")

        for i in range(self.max_iter):
            w_new, ob_heuristic = self._get_concave_minorant(r, w_prev, solver)
            true_ob, _, _ = self.utility.evaluate(w_new, r)

            if true_ob - best_ob > eps:
                best_ob = true_ob
                best_w = w_new
                w_prev = w_new
                weight_history.append(w_new)
                wall_time.append(time.time())
            else:
                wall_time.append(time.time())
                break

            if verbose:
                print(f" iteration {i} ".center(50, "#"))
                print(
                    f"{ob_heuristic=:.3f}, {true_ob=:.3f}, "
                    f"relative error: {ob_heuristic / true_ob - 1:.3f}"
                )

            if wall_time[-1] - wall_time[0] > max_time:
                wall_time.append(time.time())
                break

        if verbose:
            print(f"final utility: {true_ob:.3f}\n\n")

        self._weights = best_w
        self._weights_history = np.stack(weight_history)
        self._wall_time = np.array(wall_time)

    def _get_concave_minorant(
        self, r: np.array, w_prev: np.array, solver: Optional[str]
    ) -> Tuple[np.array, float]:
        """
        Linearizes convex terms to obtain a concave minorant of the utility
        """

        w = cp.Variable(r.shape[1], nonneg=True)

        ob, _, _ = self.utility.get_concave_minorant_expression(r, w, w_prev)

        constraints = [cp.sum(w) == 1]
        prob = cp.Problem(cp.Maximize(ob), constraints)
        prob.solve(solver=solver)

        return w.value, ob.value


class GradientOptimizer(CPTOptimizer):
    def __init__(
        self,
        utility: Union[CPTUtility, CPTUtilityGA],
        max_iter: int = 100_000,
        gpu: bool = False,
    ):

        super().__init__(utility, max_iter)
        self.device = (
            torch.device("cuda") if (torch.cuda.is_available() and gpu) else torch.device("cpu")
        )
        if isinstance(self.utility, CPTUtility):
            CPTUtilityGA.convert_to_class(self.utility)

        # Help linter
        assert isinstance(self.utility, CPTUtilityGA)
        self.utility = self.utility
        self._all_weights_history = None

    @property
    def all_weights_history(self) -> np.array:
        assert self._all_weights_history is not None
        return self._all_weights_history

    def optimize(
        self,
        returns: torch.Tensor | np.ndarray,
        verbose: bool = False,
        initial_weights: Optional[Union[torch.tensor, np.ndarray]] = None,
        starting_points: Optional[int] = None,
        lr: bool = 1e-1,
        max_time: float = np.inf,
        keep_history: bool = True,
    ) -> None:

        if isinstance(returns, np.ndarray):
            returns = torch.tensor(returns)
        returns = returns.to(self.device)

        if initial_weights is not None:
            assert not starting_points
            if isinstance(initial_weights, np.ndarray):
                if initial_weights.ndim == 1:
                    initial_weights = np.atleast_2d(initial_weights).T
                initial_weights = torch.tensor(initial_weights)
            # TODO: fix for 1d TENSOR input
        else:
            initial_weights = torch.ones(returns.shape[1], dtype=torch.float64) / returns.shape[1]
            if starting_points and starting_points > 1:
                dire = torch.distributions.dirichlet.Dirichlet(torch.ones(returns.shape[1]))
                additional_starting_weights = dire.sample([starting_points - 1])
                initial_weights = torch.vstack([initial_weights, additional_starting_weights])
            initial_weights = torch.atleast_2d(initial_weights).T

        unconstrained_w = (
            torch.log(initial_weights).to(self.device).clone().detach().requires_grad_(True)
        )
        optimizer = torch.optim.SGD([unconstrained_w], lr=lr)

        if verbose:
            print("#" * 50)
            print(" Starting GradientOptimizer ".center(50, "#"))
            print("#" * 50 + "\n\n")

        wall_time = [time.time()]
        weight_history = [initial_weights.to(self.device)]
        for i in range(self.max_iter):
            nonneg_weights = torch.exp(unconstrained_w)
            weights = nonneg_weights / nonneg_weights.sum(0, keepdim=True)  # normalize
            util = self.utility.evaluate_with_gradient(weights, returns)
            neg_util = -util
            neg_util.sum().backward()
            optimizer.step()
            optimizer.zero_grad()
            if keep_history:
                weight_history.append(weights)
            wall_time.append(time.time())

            if verbose and i % (self.max_iter // 10) == 0:
                print(f" iteration {i} ".center(50, "#"))
                print(f"best utility: {util.max():.3f}")

            if wall_time[-1] - wall_time[0] > max_time:
                break

        if verbose:
            print(f"final utility: {util.max():.3f}\n\n")

        best_weights = util.argmax()
        self._weights = weights[:, best_weights].cpu().detach().numpy()
        self._wall_time = np.array(wall_time)

        if keep_history:
            self._all_weights_history = torch.stack(weight_history).cpu().detach().numpy()
            self._weights_history = self._all_weights_history[..., best_weights]


class MeanVarianceFrontierOptimizer(CPTOptimizer):
    def optimize(
        self,
        returns: np.array,
        verbose: bool = False,
        solver: Optional[str] = None,
        samples: int = 100,
    ) -> None:

        wall_time = [time.time()]
        mu = np.mean(returns, axis=0)
        Sigma = np.cov(returns, rowvar=False)

        min_vol = np.sqrt(self._get_min_variance(Sigma))
        max_vol = np.sqrt(self._get_max_return_variance(mu))

        var_target = cp.Parameter()

        best_weights = None
        best_cpt_utility = -np.inf

        w = cp.Variable(len(mu), nonneg=True)
        objective = cp.Maximize(w @ mu)
        constraints = [cp.sum(w) == 1, cp.quad_form(w, Sigma) <= var_target]
        problem = cp.Problem(objective, constraints)

        if verbose:
            print("#" * 50)
            print(" Starting MeanVarianceOptimizer ".center(50, "#"))
            print("#" * 50 + "\n\n")

        weight_history = []
        for vol_target_val in np.linspace(min_vol, max_vol, samples):
            var_target.value = vol_target_val**2
            problem.solve()
            cpt_util = self.utility.evaluate(w.value, returns)[0]
            wall_time.append(time.time())
            weight_history.append(w.value)

            if verbose:
                print(f" volatility target {vol_target_val:.3f} ".center(50, "#"))
                print(f"utility: {cpt_util:.3f}")

            if cpt_util > best_cpt_utility:
                best_weights = w.value
                best_cpt_utility = cpt_util

        if verbose:
            print(f"final utility: {best_cpt_utility:.3f}\n\n")

        self._weights = best_weights
        self._weights_history = np.stack(weight_history)
        self._wall_time = np.array(wall_time)

    @staticmethod
    def _get_min_variance(Sigma: np.array) -> np.float:
        w = cp.Variable(Sigma.shape[0], nonneg=True)
        objective = cp.Minimize(cp.quad_form(w, Sigma))
        constraints = [cp.sum(w) == 1]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        return objective.value

    @staticmethod
    def _get_max_return_variance(mu: np.array) -> np.float:
        w = cp.Variable(len(mu), nonneg=True)
        objective = cp.Maximize(w @ mu)
        constraints = [cp.sum(w) == 1]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        return objective.value


class ConvexConcaveOptimizer(CPTOptimizer):
    def __init__(self, utility: Union[CPTUtility, CPTUtilityCC], max_iter: int = 1000):
        super().__init__(utility, max_iter)
        if isinstance(self.utility, CPTUtility):
            CPTUtilityCC.convert_to_class(self.utility)

    def optimize(
        self,
        r: np.array,
        verbose: bool = False,
        solver: Optional[str] = None,
        initial_weights: Optional[np.array] = None,
        eps: float = 1e-4,
        max_time: float = np.inf,
    ) -> None:

        w_prev = (
            initial_weights if initial_weights is not None else np.ones(r.shape[1]) / r.shape[1]
        )

        wall_time = [time.time()]
        weight_history = [w_prev]
        best_w = w_prev
        true_ob, _, _ = self.utility.evaluate(w_prev, r)
        best_ob = true_ob

        if verbose:
            print("#" * 50)
            print(" Starting ConvexConcaveOptimizer ".center(50, "#"))
            print("#" * 50 + "\n\n")

        for i in range(self.max_iter):

            trust_region = 1.0
            smallest_trust = 1e-3
            while trust_region > smallest_trust:

                pi = self.get_pi_from_w_prev(r, w_prev)
                w_new, ob_heuristic = self._get_approximate_concave_minorant(
                    r, w_prev, pi, trust_region, solver
                )
                true_ob, _, _ = self.utility.evaluate(w_new, r)

                if true_ob - best_ob > eps:
                    best_ob = true_ob
                    best_w = w_new
                    w_prev = w_new
                    break
                else:
                    trust_region /= 1.5

            if trust_region > smallest_trust:
                weight_history.append(w_new)
                wall_time.append(time.time())
            else:
                wall_time.append(time.time())
                break

            if verbose:
                print(f" iteration {i} ".center(50, "#"))
                print(
                    f"{ob_heuristic=:.3f}, {true_ob=:.3f}, "
                    f"relative error: {ob_heuristic / true_ob - 1:.3f}"
                )

            if wall_time[-1] - wall_time[0] > max_time:
                wall_time.append(time.time())
                break

        if verbose:
            print(f"final utility: {true_ob:.3f}\n\n")

        self._weights = best_w
        self._weights_history = np.stack(weight_history)
        self._wall_time = np.array(wall_time)

    def _get_approximate_concave_minorant(
        self,
        r: np.array,
        w_prev: np.array,
        pi: np.array,
        trust_region: float,
        solver: Optional[str],
    ) -> Tuple[np.array, float]:
        """
        Linearizes convex terms for a pis to obtain an approximate concave minorant of the utility
        """

        w = cp.Variable(r.shape[1], nonneg=True)

        (
            ob,
            _,
            _,
            aux_constraints,
        ) = self.utility.get_approximate_concave_minorant_expression(r, w, w_prev, pi)

        constraints = [
            cp.sum(w) == 1,
            cp.norm_inf(w - w_prev) <= trust_region,
        ] + aux_constraints
        prob = cp.Problem(cp.Maximize(ob), constraints)
        prob.solve(solver=solver)

        return w.value, ob.value

    def get_pi_from_w_prev(self, r: np.array, w_prev: np.array) -> np.array:
        N = r.shape[0]
        previous_portfolio_returns = r @ w_prev

        pos_inds = previous_portfolio_returns >= 0
        neg_inds = ~pos_inds

        pos_returns = previous_portfolio_returns[pos_inds]
        neg_returns = previous_portfolio_returns[neg_inds]

        p_weights = self.utility.cumulative_weights(N, delta=self.utility.delta_pos)[
            -len(pos_returns):
        ]
        n_weights = self.utility.cumulative_weights(N, delta=self.utility.delta_neg)[
            -len(neg_returns):
        ]

        p_weights_sorted = p_weights[pos_returns.argsort().argsort()]
        n_weights_sorted = n_weights[(-neg_returns).argsort().argsort()]

        pi = np.zeros(N)
        pi[pos_inds] = p_weights_sorted
        pi[neg_inds] = n_weights_sorted
        return pi

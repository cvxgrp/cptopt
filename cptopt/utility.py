from typing import Union, Tuple, List, Any

import cvxpy as cp
import numpy as np
import torch


class CPTUtility:
    """
    A utility function that incorporates features from cumulative prospect theory.
    1. Prospect theory utility ('S-shaped'), parametrized by gamma_pos and gamma_neg
    2. Overweighting of extreme outcomes, parametrized by delta_pos and delta_neg
    """

    def __init__(self, gamma_pos: float, gamma_neg: float, delta_pos: float, delta_neg: float):

        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg

        self.delta_pos = delta_pos
        self.delta_neg = delta_neg

        self._validate_arguments()

    @staticmethod
    def _weight(p: np.array, delta: float) -> np.array:
        assert delta >= 0.278, (
            f"[utility] weights are only strictly increasing for delta >= 0.278."
            f"{delta=} was passed."
        )
        return (p**delta) / ((p**delta + np.maximum((1 - p), 0) ** delta) ** (1 / delta))

    def cumulative_weights(self, N: int, delta: float) -> np.array:

        pi = -np.diff(self._weight(np.flip(np.cumsum(np.ones(N) / N)), delta))
        pi = np.append(pi, np.array([self._weight(1 / N, delta)]))

        # make monotone
        assert np.sum(np.diff(np.diff(pi) > 0)) <= 1, "[utility] probabilities should be unimodal."
        idx_min = np.argmin(pi)
        pi[:idx_min] = pi[idx_min]
        return pi

    def evaluate(self, weights: np.array, returns: np.array) -> Tuple[float, float, float]:
        portfolio_returns = returns @ weights
        N = len(portfolio_returns)

        p_weights = self.cumulative_weights(N, delta=self.delta_pos)
        n_weights = self.cumulative_weights(N, delta=self.delta_neg)

        pos_sort = np.sort(np.maximum(portfolio_returns, 0))
        util_p = p_weights @ self.p_util_expression(pos_sort).value

        neg_sort = np.flip(np.sort(np.minimum(portfolio_returns, 0)))
        util_n = n_weights @ self.n_util(neg_sort)

        return util_p - util_n, util_p, util_n

    def _validate_arguments(self) -> None:
        assert self.gamma_neg >= self.gamma_pos > 0, (
            f"[utility] Loss aversion implies gamma_neg >= gamma_pos. "
            f"Here: {self.gamma_neg=}, {self.gamma_pos=}."
        )
        assert self.delta_pos > 0, f"[utility] delta_pos must be positive: {self.delta_pos=}."
        assert self.delta_neg > 0, f"[utility] delta_neg must be positive: {self.delta_neg=}."

    def p_util_expression(self, portfolio_returns: Union[np.array, cp.Expression]) -> cp.Expression:
        return 1 - cp.exp(-self.gamma_pos * portfolio_returns)

    def n_util(self, portfolio_returns: np.array) -> np.array:
        return 1 - np.exp(self.gamma_neg * portfolio_returns)


class ConverterMixin:
    @classmethod
    def convert_to_class(cls: Any, obj: Any) -> None:
        obj.__class__ = cls


class CPTUtilityGA(ConverterMixin, CPTUtility):
    @staticmethod
    def utility_with_gradient(portfolio_returns: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        return 1 - torch.exp(-gamma * portfolio_returns)

    def evaluate_with_gradient(self, weights: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        device = weights.device

        R = returns @ weights
        N = R.shape[0]

        pos_sort = torch.sort(torch.maximum(R, torch.tensor(0, device=device)), axis=0)[0]
        p_weights = torch.tensor(self.cumulative_weights(N, delta=self.delta_pos), device=device)

        positive_contribution = p_weights @ self.utility_with_gradient(
            pos_sort, gamma=torch.tensor(self.gamma_pos, device=device)
        )

        neg_sort = torch.sort(-torch.minimum(R, torch.tensor(0, device=device)), axis=0)[0]
        n_weights = torch.tensor(self.cumulative_weights(N, delta=self.delta_neg), device=device)

        negative_contribution = n_weights @ self.utility_with_gradient(
            neg_sort, gamma=torch.tensor(self.gamma_neg, device=device)
        )

        return positive_contribution - negative_contribution


class CPTUtilityMM(ConverterMixin, CPTUtility):
    def get_concave_minorant_expression(
        self,
        returns: np.ndarray,
        new_weights: cp.Variable,
        previous_weights: np.ndarray,
    ) -> Tuple[cp.Expression, cp.Expression, cp.Expression]:
        N = returns.shape[0]
        previous_portfolio_returns = returns @ previous_weights
        new_returns = returns @ new_weights

        p_weights = self.cumulative_weights(N, delta=self.delta_pos)
        n_weights = self.cumulative_weights(N, delta=self.delta_neg)

        lin_neg_util = self.linearize_neg_utility(new_returns, previous_portfolio_returns)
        negative_contributions = cp.dotsort(cp.pos(lin_neg_util), n_weights)

        previous_utility = self.p_util_expression(previous_portfolio_returns).value
        new_utility = self.p_util_expression(new_returns)
        positive_contributions = self.get_linearize_weighted_pos_expression(
            new_utility, previous_utility, p_weights
        )

        ob = positive_contributions - negative_contributions
        return ob, positive_contributions, negative_contributions

    @staticmethod
    def get_linearize_weighted_pos_expression(
        new_utility: cp.Expression, previous_utility: np.array, weights: np.array
    ) -> cp.Expression:
        assert (np.diff(weights) >= -1e-5).all()
        x_0_inds = np.argsort(previous_utility)
        x_0_sorted = previous_utility[x_0_inds]
        x_sorted = new_utility[x_0_inds]
        grad_top_k_pos = weights * (x_0_sorted >= 0)

        weighted_pos_prev = cp.dotsort(cp.pos(previous_utility), weights).value
        return weighted_pos_prev + grad_top_k_pos @ (x_sorted - x_0_sorted)

    def linearize_neg_utility(
        self,
        current_portfolio_returns: cp.Expression,
        previous_portfolio_returns: np.array,
    ) -> cp.Expression:
        """
        Linearizes the negative utility 1-exp(gn * r)
        """
        grad_neg_util = -self.gamma_neg * np.exp(self.gamma_neg * previous_portfolio_returns)
        prev_neg_utils = self.n_util(previous_portfolio_returns)
        return prev_neg_utils + cp.multiply(
            grad_neg_util, (current_portfolio_returns - previous_portfolio_returns)
        )


class CPTUtilityCC(ConverterMixin, CPTUtility):
    def get_approximate_concave_minorant_expression(
        self,
        returns: np.ndarray,
        new_weights: cp.Variable,
        previous_weights: np.ndarray,
        pi: np.ndarray,
    ) -> Tuple[cp.Expression, cp.Expression, cp.Expression, List[cp.Expression]]:
        previous_portfolio_returns = returns @ previous_weights
        new_returns = returns @ new_weights

        lin_neg_utilities = self.linearize_convex_concave_negative(
            new_returns, previous_portfolio_returns
        )
        negative_contributions = pi @ lin_neg_utilities

        pos_utilities, aux_constraints = self.convex_concave_positive(new_returns)
        positive_contributions = pi @ pos_utilities

        ob = positive_contributions + negative_contributions
        return ob, positive_contributions, negative_contributions, aux_constraints

    def linearize_convex_concave_negative(
        self, new_returns: cp.Expression, previous_portfolio_returns: np.array
    ) -> cp.Expression:
        grad_conv = (
            self.gamma_neg * np.exp(self.gamma_neg * previous_portfolio_returns) - self.gamma_neg
        )
        grad_conv[previous_portfolio_returns >= 0] = 0
        prev_conv = (
            -1
            + np.exp(self.gamma_neg * previous_portfolio_returns)
            - self.gamma_neg * previous_portfolio_returns
        )
        prev_conv[previous_portfolio_returns >= 0] = 0
        return prev_conv + cp.multiply(grad_conv, (new_returns - previous_portfolio_returns))

    def convex_concave_positive(
        self, new_returns: cp.Expression
    ) -> Tuple[cp.Variable, List[cp.Expression]]:
        N = new_returns.shape[0]
        r_neg = cp.Variable(N, nonpos=True)
        r_pos = cp.Variable(N, nonneg=True)

        t = cp.Variable(N)
        t1 = cp.Variable(N)
        t2 = cp.Variable(N)

        extended = self.gamma_neg * r_neg
        util = 1 - cp.exp(-self.gamma_pos * r_pos)

        aux_constraints = [
            t <= t1 + t2,
            new_returns == r_neg + r_pos,
            t1 == extended,
            t2 <= util,
        ]
        return t, aux_constraints

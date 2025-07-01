# pricing.py
from abc import ABC, abstractmethod
import numpy as np

class PricingStrategy(ABC):
    @abstractmethod
    def reset(self, params: list, env) -> None:
        """
        Called once at the start of an episode with the PSO-found parameters.
        Params is a flat list of floats.
        """
        pass

    @abstractmethod
    def get_tolls(self, env, t: int) -> np.ndarray:
        """
        Return an array of shape (n_links,) of tolls at time t.
        """
        pass


class LinearPricing(PricingStrategy):
    def reset(self, params, env):
        # Expect params = [a0, b0, a1, b1, …] of length 2 * n_links
        self.n = len(env.possible_agents)
        arr = np.array(params)
        self.a = arr[0::2][:self.n]
        self.b = arr[1::2][:self.n]
        self.lb = env.price_lower_bound
        self.ub = env.price_upper_bound

    def get_tolls(self, env, t):
        # env.flows should be current demand per link
        demand = env.flows
        tolls = self.a * demand + self.b
        return np.clip(tolls, self.lb, self.ub)


class FixedPricing(PricingStrategy):
    def reset(self, params, env):
        # Expect params = [p0, p1, …] of length n_links
        self.tolls = np.clip(np.array(params),
                             env.price_lower_bound,
                             env.price_upper_bound)

    def get_tolls(self, env, t):
        return self.tolls


class UnboundPricing(PricingStrategy):
    def reset(self, params, env):
        # Expect params = [p_0,0, p_0,1, …, p_T-1,L-1] of length timesteps * n_links
        self.T = env.timesteps
        self.n = len(env.possible_agents)
        self.lb = env.price_lower_bound
        self.ub = env.price_upper_bound
        if not params:
            self.schedule = np.zeros((self.T, self.n))
            return
            # otherwise proceed as normal
        self.schedule = np.array(params).reshape(self.T, self.n)


    def get_tolls(self, env, t):
        # Clamp time index to valid range [0, T-1]
        idx = t if t < self.schedule.shape[0] else self.schedule.shape[0] - 1
        return np.clip(self.schedule[idx], self.lb, self.ub)
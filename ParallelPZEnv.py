from pettingzoo import ParallelEnv
import functools
import random
from collections import Counter, defaultdict, deque
from functools import partial
from random import choices, randint

import numpy as np

from pettingzoo import ParallelEnv
from gymnasium.spaces import Discrete, Box

import numpy.random as nprand

from RLUtils import quick_get_new_travel_times
from TimeOnlyUtils import QueueRanges, volume_delay_function

n_cars = 850
n_timesteps = 1000
# timeseed = 0
# votseed = 0
NONE = 3


class simulation_env(ParallelEnv):
    # metadata = {'is_parallelizable': True}
    metadata = {'name': 'MMRP Simulation'}
    def __init__(self, render_mode=None):
        self.num_routes = 2
        self.possible_agents = ['route_' + str(r) for r in range(self.num_routes)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.render_mode = None
        # self.action_spaces = {agent: Discrete(3) for agent in self.possible_agents}

        # parameters that should mostly stay the same. if they need changing, changing in here should edit all exps
        self.timesteps = n_timesteps
        self.beta_dist_alpha = 5
        self.beta_dist_beta = 5
        self.n_cars = n_cars

        self.car_vot_upperbound = 0.999
        self.car_vot_lowerbound = 0.001
        # self.car_vot_upperbound = 9.5
        # self.car_vot_lowerbound = 2.5
        self.bound = 0.25
        self.pricing_dict = {
            -1: lambda x: max(x - self.bound, self.bound),
            0: lambda x: x,
            1: lambda x: x + self.bound,
        }

        self.queues_manager = QueueRanges()

        # self.agent_reward_norms = {agent: [] for agent in range(self.num_routes)}
        self.agent_reward_norms_lens = {agent: None for agent in range(self.num_routes)}
        self.agent_reward_norms_mean = {agent: None for agent in range(self.num_routes)}
        self.agent_reward_norms_vars = {agent: None for agent in range(self.num_routes)}

        self.agent_vdf_cache = {agent: {} for agent in range(self.num_routes)}

        self.agent_price_range = {agent: None for agent in range(self.num_routes)}
        self.agent_maxes = {agent: None for agent in range(self.num_routes)}
        self.agent_mins = {agent: None for agent in range(self.num_routes)}


    def update_rolling_norms(self, agent, new_value):
        # n_old = len(self.agent_reward_norms[agent]) # the lowest this can be is 0
        # if n_old == 0:
        n_old = self.agent_reward_norms_lens[agent] if self.agent_reward_norms_lens[agent] is not None else 0
        if self.agent_reward_norms_mean[agent] is None or self.agent_reward_norms_vars[agent] is None:
            # self.agent_reward_norms[agent] = self.agent_reward_norms[agent] + [new_value]
            self.agent_reward_norms_mean[agent] = new_value
            self.agent_reward_norms_vars[agent] = 1
            self.agent_reward_norms_lens[agent] = 1
            return self.agent_reward_norms_mean[agent],  self.agent_reward_norms_vars[agent]
        else:
            new_mean = (((self.agent_reward_norms_mean[agent] * n_old) + new_value)/(n_old + 1))
            new_var = (((n_old-1)/(n_old))*self.agent_reward_norms_vars[agent]) + ((new_value - self.agent_reward_norms_mean[agent]) ** 2)/(n_old+1)
            # self.agent_reward_norms[agent] = self.agent_reward_norms[agent] + [new_value]
            self.agent_reward_norms_mean[agent] = new_mean
            self.agent_reward_norms_vars[agent] = new_var
            self.agent_reward_norms_lens[agent] += 1
            return self.agent_reward_norms_mean[agent], self.agent_reward_norms_vars[agent]


    def quantalify(self, r, rest, lambd=0.9):
        # breakpoint()
        return np.exp(lambd * r) / np.sum([np.exp(lambd * re) for re in rest], axis=-0)

    def new_quantal_decision(self, routes):
        utility = [u[1] for u in routes]
        utility = [u - max(utility) for u in utility]
        quantal_weights = [
            self.quantalify(x, np.asarray(utility, dtype=np.float32)) for x in utility
        ]
        choice = choices(routes, weights=quantal_weights)
        # print("101:", [q/sum(quantal_weights) for q in quantal_weights], utility)
        return choice[0]

    # def quantal_decision(self, routes):
    #     # We pass in a list of 2-tuple - (road, utility) for each road.
    #     utility = [u[1] for u in routes]
    #     utility = [u - max(utility) for u in utility]
    #     quantal_weights = shortform_quantal_function(utility)
    #     choice = choices(routes, weights=quantal_weights)
    #     print(self.time, "::INF:: ", [u[1] for u in routes], quantal_weights, choice[0])
    #     # print("101:", [q/sum(quantal_weights) for q in quantal_weights], utility)
    #     return choice[0]

    def generate_car_time_distribution(self, timeseed=None):
        if timeseed is not None:
            nprand.seed(timeseed)
        car_dist_norm = nprand.beta(
            self.beta_dist_alpha,
            self.beta_dist_beta,
            size=self.n_cars,
        )
        car_dist_arrival = list(
            map(
                lambda z: round(
                    (z - min(car_dist_norm))
                    / (max(car_dist_norm) - min(car_dist_norm))
                    * self.timesteps
                ),
                car_dist_norm,
            )
        )
        if timeseed is not None:
            nprand.seed(None)
        return car_dist_arrival

    def generate_car_vot_distribution(self, votseed=None):
        if votseed is not None:
            nprand.seed(votseed)
        car_vot = nprand.uniform(
            self.car_vot_lowerbound, self.car_vot_upperbound, self.n_cars
        )
        if votseed is not None:
            nprand.seed(None)
        return car_vot

    def is_simulation_complete(self):
        if self.time > self.timesteps + 1:
            return True
        else:
            return False

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(low=np.array([0, 0, 0, 15, 0, 0, 15]), high=np.array([self.n_cars, 200, self.n_cars, 1000000, self.n_cars, 200, 1000000]), dtype=np.float64)
        # return self.observation_space_d[agent]

    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(3)

    def reset(self, seed=None, options=None, set_np_seed=None):
        # This is the stuff that PettingZoo needs
        self.agents = self.possible_agents[:]

        # self.rewards = {agent: 0 for agent in self.agents}
        # self._cumulative_rewards = {agent: 0 for agent in self.agents}
        # self.terminations = {agent: False for agent in self.agents}
        # self.truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        observations = {agent: np.array([0, 0, 0, 15, 0, 0, 15]) for agent in self.agents}

        self.state = observations
        self.num_moves = 0

        self.car_dist_arrival = self.generate_car_time_distribution(timeseed=np.random.randint(51))
        self.car_vot_arrival = self.generate_car_vot_distribution(votseed=np.random.randint(51))
        self.time = 0
        self.roadQueues = {r: [] for r in range(self.num_routes)}
        self.roadVDFS = {
            0: partial(volume_delay_function, 0.656, 4.8, 15, 20),
            1: partial(volume_delay_function, 0.656, 4.8, 30, 20),
        }
        self.roadTravelTime = {r: self.roadVDFS[r](0) for r in self.roadVDFS.keys()}
        self.arrival_timestep_dict = Counter(self.car_dist_arrival)
        # self.roadPrices = {r: 20.0 for r in self.roadVDFS.keys()}
        self.roadPrices = {0: randint(1,40), 1: randint(1,40)}

        self.agent_price_range = {agent: 0 for agent in range(self.num_routes)}
        self.agent_maxes = {agt: price for agt, price in self.roadPrices.items()}
        self.agent_mins = {agt: price for agt, price in self.roadPrices.items()}

        self.queues_manager.reset()

        self.time_out_car = {r: defaultdict(int) for r in self.roadVDFS.keys()}
        self.arrived_vehicles = []
        self.vot_deque = deque(self.car_vot_arrival)

        # print(self.observations)
        return observations, infos
        # return observations



    def get_observe(self, agent):
        agent_id = self.agent_name_mapping[agent]
        not_agent_id = [x for x in self.agent_name_mapping.values() if x is not agent_id]
        observed = [
            len(self.roadQueues[agent_id]),
            self.roadPrices[agent_id],
            self.arrival_timestep_dict[self.time],
            self.roadTravelTime[agent_id]
        ] + [
            len(self.roadQueues[agt]) for agt in not_agent_id
        ] + [
            self.roadPrices[agt] for agt in not_agent_id
        ] + [
            self.roadTravelTime[agt] for agt in not_agent_id
        ]
        cars = self.arrived_vehicles

        # TODO: FINISH THIS
        return np.array(observed, dtype=np.float32)

    def close(self):
        pass

    def get_utility(self, travel_time, econom_cost, vot):
        return -((vot * travel_time) + econom_cost)

    def generate_utility_funct(self, travel_time, econom_cost):
        return partial(self.get_utility, travel_time, econom_cost)

    def step(self, actions):
        if not actions:
            self.agents = []
            return ({} for _ in range(self.num_routes))

        # update the price of the road
        for agent_name, action in actions.items():
            agent_id = self.agent_name_mapping[agent_name]
            action = action - 1

            # update the price ranges, used in the debugging/tracking
            self.roadPrices[agent_id] = self.pricing_dict[action](self.roadPrices[agent_id])
            if self.roadPrices[agent_id] > self.agent_maxes[agent_id]:
                self.agent_maxes[agent_id] = self.roadPrices[agent_id]
                self.agent_price_range[agent_id] = self.agent_maxes[agent_id] - self.agent_mins[agent_id]
            if self.roadPrices[agent_id] < self.agent_mins[agent_id]:
                self.agent_mins[agent_id] = self.roadPrices[agent_id]
                self.agent_price_range[agent_id] = self.agent_maxes[agent_id] - self.agent_mins[agent_id]

        # Here, we need to update the simulation and push forward with one timestep
        # Once we've updated all of that, we then update the rewards
        # first, we update the travel time, the queues and the arrived vehicles
        self.roadTravelTime, self.roadQueues, self.arrived_vehicles, self.queues_manager, self.agent_vdf_cache = quick_get_new_travel_times(
            self.roadQueues, self.roadVDFS, self.time, self.arrived_vehicles, self.time_out_car, self.queues_manager, self.agent_vdf_cache
        )
        # next, we update the car arrivals
        num_vehicles_arrived = self.arrival_timestep_dict[self.time]
        cars_arrived_vot = [self.vot_deque.popleft() for _ in range(num_vehicles_arrived)]

        # we generate the utility function for each road
        road_partial_funct = {
            r: self.generate_utility_funct(self.roadTravelTime[r], self.roadPrices[r])
            for r in self.roadVDFS.keys()
        }
        car_utilities = {
            n: list(
                {
                    r: utility_funct(n_car_vot)
                    for r, utility_funct in road_partial_funct.items()
                }.items()
            )
            for n, n_car_vot in enumerate(cars_arrived_vot)
        }

        car_quantal_decision = {
            c: self.new_quantal_decision(r) for c, r in car_utilities.items()
        }

        decisions = zip([d[0] for d in car_quantal_decision.values()], cars_arrived_vot)
        timestep_rewards = {agt: 0 for agt in self.roadVDFS.keys()}
        for decision, vot in decisions:
            self.roadQueues[decision] = self.roadQueues[decision] + [
                (
                    decision,
                    self.time,
                    vot,
                    self.time + self.roadTravelTime[decision],
                    self.roadPrices[decision],
                )
            ]


            # Uncomment below for maximising profit
            # timestep_rewards[decision] = timestep_rewards[decision] + self.roadPrices[decision]
            # Uncomment bellow for minimising total combined cost
            # timestep_rewards[decision] = timestep_rewards[decision] - ((vot*self.roadTravelTime[decision]) + self.roadPrices[decision])

            self.time_out_car[decision][round(self.time + self.roadTravelTime[decision])] = (
                    self.time_out_car[decision][round(self.time + self.roadTravelTime[decision])] + 1
            )
        # Uncomment below for minimising travel time
        norm_rewards = {}
        for agent in self.roadVDFS.keys():
            # self.agent_reward_norms[agent] = self.agent_reward_norms[agent] + [timestep_rewards[agent]]
            # agent_reward_mean = np.mean(self.agent_reward_norms[agent])
            # agent_reward_std = np.std(self.agent_reward_norms[agent])
            timestep_rewards[agent] = -self.roadTravelTime[agent]
            agent_reward_mean, agent_reward_var = self.update_rolling_norms(agent, timestep_rewards[agent])
            agent_reward_std = np.sqrt(agent_reward_var)
            norm_timestep_reward = (timestep_rewards[agent] - agent_reward_mean)/(1 if agent_reward_std == 0 else agent_reward_std)
            norm_rewards["route_" + str(agent)] = np.float32(norm_timestep_reward)
            # norm_rewards["route_" + str(agent)] = timestep_rewards[agent]
        # timestep_rewards = {"route_" + str(a): -x for a, x in self.roadTravelTime.items()}
        # print(norm_rewards)
        """
        UNCOMMENT FOR PROFIT (UNCOMMENT LINE 250 AS WELL, MAKE SURE TO ADD 1 INSTEAD OF REMOVING THE COMBINED COST)
        """
        # if num_vehicles_arrived > 0:
        #     timestep_rewards = {a: x*self.roadPrices[a] for a,x in timestep_rewards.items()}
        #     # timestep_rewards = {"route_"+str(a): x/sum(timestep_rewards.values()) for a,x in timestep_rewards.items()}
        #     timestep_rewards_norm = scale([r for r in timestep_rewards.values()])
        #     timestep_rewards = {"route_" + str(a): x for a, x in zip(timestep_rewards.keys(), timestep_rewards_norm)}
        # else:
        #     timestep_rewards = {"route_" + str(a): 0 for a in self.roadVDFS.keys()}

        # Scale the rewards
        # timestep_rewards_norm = scale([r for r in timestep_rewards.values()])
        # timestep_rewards = {"route_" + str(a): x for a, x in zip(timestep_rewards.keys(), timestep_rewards_norm)}
        # print(self.time, "rewards:", norm_rewards, ", Travel time:", self.roadTravelTime, ", Prices:",self.roadPrices, ", N_queue:",{r: len(x) for r, x in self.roadQueues.items()}, ", N cars arrived:",num_vehicles_arrived)
        observations = {agent: self.get_observe(agent) for agent in self.agents}
        rewards = norm_rewards
        terminations = {"route_" + str(a): False for a in self.roadVDFS.keys()}
        truncations = {"route_" + str(a): False for a in self.roadVDFS.keys()}
        infos = {agent: {} for agent in self.agents}
        self.time += 1
        if self.is_simulation_complete():
            self.agents = []
            terminations = {"route_" + str(a): True for a in self.roadVDFS.keys()}
            for road, queue in self.roadQueues.items():
                self.arrived_vehicles = self.arrived_vehicles + [car for car in self.roadQueues[road]]

            self.travel_time = [(c[3] - c[1]) for c in self.arrived_vehicles]
            # print(np.mean(self.travel_time))
            self.time_cost_burden = [((c[3] - c[1]) * c[2]) for c in self.arrived_vehicles]
            self.combined_cost = [((c[3] - c[1]) * c[2]) + c[4] for c in self.arrived_vehicles]
            self.road_profits = {r: sum([x[4] for x in self.arrived_vehicles if x[0] == r]) for r in self.roadQueues.keys()}
            # print("END OF SIM ARRIVED CARS", len(self.arrived_vehicles))
        return observations, rewards, terminations, truncations, infos




        # else:
        #     self.state[self.agents[1 - self.agent_name_mapping[agent]]] = NONE
        #     self._clear_rewards()
        # self.agent_selection = self._agent_selector.next()
        # self._accumulate_rewards()


import math
from queue import PriorityQueue

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
# from wandb import agent

# from RLUtils import quick_get_new_travel_times
# from RLUtils import quicker_get_new_travel_times as quick_get_new_travel_times
from TimeOnlyUtils import QueueRanges, volume_delay_function
from TravelTimeRLUtils import pq_get_new_travel_times

n_cars = 850
n_timesteps = 1000
# timeseed = 0
# votseed = 0
NONE = 3


class simulation_env(ParallelEnv):
    # metadata = {'is_parallelizable': True}
    metadata = {"name": "MMRP Simulation"}

    def __init__(
        self,
        render_mode=None,
        num_routes=2,
        initial_road_cost="Fixed",
        fixed_road_cost=20.0,
        arrival_dist="Beta",
        normalised_obs=False,
        road0_capacity=15,
        road0_fftraveltime=20,
        road1_capacity=30,
        road1_fftraveltime=20,
        reward_fn = "MinVehTravelTime",
        in_n_cars=n_cars,
        road_vdfs=None,
        free_roads=None, #TODO: finish implementing this
    ):
        """
        Params:
        render_mode: Does nothing.

        initial_road_cost:  Can currently be 'Fixed' or 'Random'. When random, we bound between 10 and 40. When fixed,
                            we set the initial road price to the value of 'fixed_road_cost'.

        fixed_road_cost:    Only used when initial_road_cost is set to 'Fixed'. See initial_road_cost for details.

        arrival_dist:       Can be set to 'linear' or 'beta' and determines how the arrival time of users is generated.
                            Beta will generate a peak for the arrivals whereas linear will use a uniform distribution.

        normalised_obs:     Determines whether we provide the raw values from observations to the agent or if we
                            normalise the values to be between 0-1.

        roadX_capacity=15:      capacity of road X
        roadX_fftraveltime=20:  free flow travel time for road X
                                These options currently exist for road0 and road1

        reward_fn:          reward function used to calculate agent reward.
                            currently: [MaxProfit, MinVehTravelTime, MinCombinedCost]
        """
        self.num_routes = num_routes
        if self.num_routes > 2 and road_vdfs is None:
            raise Exception("More than 2 routes defined, but no VDFs provided for them. Fix this and re-run.")
        self.possible_agents = ["route_" + str(r) for r in range(self.num_routes)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        self.default_road_vdfs = road_vdfs

        self.render_mode = None
        self.initial_road_cost = initial_road_cost
        self.fixed_road_cost = fixed_road_cost
        self.arrival_dist = arrival_dist
        self.normalised_obs = normalised_obs
        self.road0_capacity = road0_capacity
        self.road0_fftraveltime = road0_fftraveltime
        self.road1_capacity = road1_capacity
        self.road1_fftraveltime = road1_fftraveltime
        self.reward_fn = reward_fn

        #TODO: finish implementing this
        self.free_roads = free_roads

        # parameters that should mostly stay the same. if they need changing, changing in here should edit all exps
        self.timesteps = n_timesteps
        self.beta_dist_alpha = 5
        self.beta_dist_beta = 5
        if in_n_cars is None:
            self.n_cars = n_cars
        else:
            self.n_cars = in_n_cars

        self.actions = None

        self.car_vot_upperbound = 0.999
        self.car_vot_lowerbound = 0.001

        self.bound = 1
        self.price_lower_bound = self.bound
        # self.price_lower_bound = 0
        # self.price_upper_bound = math.floor((self.timesteps * self.bound)/2)
        self.price_upper_bound = 125

        # self.car_vot_upperbound = 9.5
        # self.car_vot_lowerbound = 2.5
        self.pricing_dict = {
            -1: lambda x: max(x - self.bound, self.bound),
            0: lambda x: x,
            1: lambda x: min(x + self.bound, self.price_upper_bound),
        }


        # if normalised_obs:
        if self.num_routes == 2 and road_vdfs is None:
            self.max_road_travel_time = [
                volume_delay_function(
                    None, None, self.road0_capacity, self.road0_fftraveltime, n_cars
                ),
                volume_delay_function(
                    None, None, self.road1_capacity, self.road1_fftraveltime, n_cars
                ),
            ]
        else:
            self.max_road_travel_time = [
                vdf(n_cars) for vdf in self.default_road_vdfs
            ]

        self.queues_manager = QueueRanges(self.num_routes)
        #
        # self.agent_reward_norms_lens = {agent: None for agent in range(self.num_routes)}
        # self.agent_reward_norms_mean = {agent: None for agent in range(self.num_routes)}
        # self.agent_reward_norms_vars = {agent: None for agent in range(self.num_routes)}

        self.agent_vdf_cache = {agent: {} for agent in range(self.num_routes)}

        self.agent_price_range = {agent: None for agent in range(self.num_routes)}
        self.agent_maxes = {agent: None for agent in range(self.num_routes)}
        self.agent_mins = {agent: None for agent in range(self.num_routes)}
        self.agent_prices = {agent: None for agent in range(self.num_routes)}

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

    def generate_car_time_distribution(self, timeseed=None):
        if timeseed is not None:
            nprand.seed(timeseed)
        if self.arrival_dist == "Beta":
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
        elif self.arrival_dist == "Linear":
            car_dist_arrival = [round(x) for x in nprand.uniform(
                low=0, high=self.timesteps, size=self.n_cars
            )]


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
        if self.time > self.timesteps:
            return True
        else:
            return False

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """
        Observations:
        1. number of cars in this road queue
        2. price of road
        3. number of cars arriving at this timestep
        4. current travel time of the road
        5. number of cars on other road queue
        6. price of other road
        7. road travel time for other road

        8. current timestep
        9. this agent's previous action
        10. other agents previous action

        11. number of vehicles left to arrive
        12. number of vehicles on road or at destination

        """
        if self.normalised_obs:
            return Box(
                low=np.array([0 for _ in range(8+(4*(self.num_routes-1)))]),
                high=np.array([1.0 for _ in range(8)] + [2, 2] + [1.0 for _ in range(2)]),
                dtype=np.float64,
            )
        else:
            return Box(
                low=np.array([0 for _ in range(8+(4*(self.num_routes-1)))]),
                high=np.array(
                    [
                        self.n_cars,
                        40 + (n_timesteps * self.bound),
                        self.n_cars,
                        self.max_road_travel_time[1],
                        *tuple(self.n_cars for _ in range(self.num_routes - 1)),
                        *tuple((40 + (n_timesteps * self.bound) for _ in range(self.num_routes -1))),
                        *tuple(self.max_road_travel_time[1] for _ in range(self.num_routes -1)),
                        n_timesteps + 1,
                        2,
                        *tuple(2 for _ in range(self.num_routes - 1)),
                        self.n_cars,
                        self.n_cars,
                    ]
                ),
                dtype=np.float64,
            )
        # return self.observation_space_d[agent]

    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(3)

    def reset(self, seed=None, options=None, set_np_seed=None, random_cars=False, set_cars=None):
        # This is the stuff that PettingZoo needs
        self.agents = self.possible_agents[:]

        # self.rewards = {agent: 0 for agent in self.agents}
        # self._cumulative_rewards = {agent: 0 for agent in self.agents}
        # self.terminations = {agent: False for agent in self.agents}
        # self.truncations = {agent: False for agent in self.agents}
        if self.normalised_obs:
            observations = {
                agent: np.array([0 for _ in range(8+(4*(self.num_agents-1)))]) for agent in self.agents
            }
        else:
            observations = {
                # agent: np.array([0, 0, 0, 15, 0, 0, 15, 0, 0, 0, 0, 0]) for agent in self.agents
                agent: np.array([0 for _ in range(8 + (4 * (self.num_agents - 1)))]) for agent in self.agents
            }

        self.state = observations
        self.num_moves = 0

        if set_cars is not None:
            self.n_cars = set_cars
        elif random_cars:
            self.n_cars = randint(5000, 15000)

        self.car_dist_arrival = self.generate_car_time_distribution(
            timeseed=set_np_seed
        )
        self.car_vot_arrival = self.generate_car_vot_distribution(
            votseed=set_np_seed
        )
        self.time = 0
        self.roadQueues = {r: [] for r in range(self.num_routes)}

        if self.num_routes == 2 and self.default_road_vdfs is None:
            # self.roadVDFs = self.default_road_vdfs
            self.roadVDFS = {
                0: partial(
                    volume_delay_function,
                    0.656,
                    4.8,
                    self.road0_capacity,
                    self.road0_fftraveltime,
                ),
                1: partial(
                    volume_delay_function,
                    0.656,
                    4.8,
                    self.road1_capacity,
                    self.road1_fftraveltime,
                ),
            }
        else:
            self.roadVDFS = {x: vdf for x, vdf in enumerate(self.default_road_vdfs)}


        self.roadTravelTime = {r: self.roadVDFS[r](0) for r in self.roadVDFS.keys()}
        self.arrival_timestep_dict = Counter(self.car_dist_arrival)

        self.max_number_arrived_cars = max(self.arrival_timestep_dict.values())

        if self.initial_road_cost == "Fixed":
            self.roadPrices = {r: self.fixed_road_cost for r in self.roadVDFS.keys()}
        elif self.initial_road_cost == "Random":
            self.roadPrices = {0: randint(1, 40), 1: randint(1, 40)}

        infos = {agent: {'action_mask':
            [
                False if self.roadPrices[self.agent_name_mapping[agent]] <= self.price_lower_bound else True,
                True,
                False if self.roadPrices[self.agent_name_mapping[agent]] >= self.price_upper_bound else True
            ]
        }
            for agent in self.agents
        }

        self.agent_price_range = {agent: 0 for agent in range(self.num_routes)}
        self.agent_maxes = {agt: price for agt, price in self.roadPrices.items()}
        self.agent_mins = {agt: price for agt, price in self.roadPrices.items()}
        self.agent_prices = {agt: [price] for agt, price in self.roadPrices.items()}
        self.queues_manager.reset()

        self.time_out_car = {r: defaultdict(int) for r in self.roadVDFS.keys()}
        self.time_out_pq = {r: PriorityQueue() for r in self.roadVDFS.keys()}
        self.arrived_vehicles = []
        self.vot_deque = deque(self.car_vot_arrival)

        # print(self.observations)
        return observations, infos
        # return observations

    def get_observe(self, agent):
        agent_id = self.agent_name_mapping[agent]
        not_agent_id = [
            x for x in self.agent_name_mapping.values() if x is not agent_id
        ]
        if self.normalised_obs == False:
            observed = (
                [
                    len(self.roadQueues[agent_id]),
                    self.roadPrices[agent_id],
                    self.arrival_timestep_dict[self.time],
                    self.roadTravelTime[agent_id],
                ]
                + [len(self.roadQueues[agt]) for agt in not_agent_id]
                + [self.roadPrices[agt] for agt in not_agent_id]
                + [self.roadTravelTime[agt] for agt in not_agent_id]
                + [self.time]
                + [self.actions['route_' + str(agent_id)] if self.actions is not None else 1]
                + [(self.actions['route_' + str(n_agt)] if self.actions is not None else 1) for n_agt in not_agent_id]
                + [len(self.arrived_vehicles)]
                + [self.n_cars - len(self.arrived_vehicles)]
            )
        elif self.normalised_obs == True:
            observed = (
                    [
                        len(self.roadQueues[agent_id]) / (n_cars/2),
                        (self.roadPrices[agent_id] - 0.25) / ((40 + (n_timesteps * self.bound))-0.25),
                        self.arrival_timestep_dict[self.time] / self.max_number_arrived_cars,
                        (self.roadTravelTime[agent_id] - (self.road0_fftraveltime if agent_id == 0 else self.road1_fftraveltime)) / self.max_road_travel_time[agent_id],
                    ]
                    + [len(self.roadQueues[agt]) / (n_cars/2) for agt in not_agent_id]
                    + [(self.roadPrices[agt] - 0.25) / ((40 + (n_timesteps * self.bound))-0.25) for agt in not_agent_id]
                    + [(self.roadTravelTime[agt] - (self.road0_fftraveltime if agt == 0 else self.road1_fftraveltime)) / self.max_road_travel_time[agt] for agt in not_agent_id]
                    + [self.time / n_timesteps]
                    + [self.actions['route_' + str(agent_id)] if self.actions is not None else 1]
                    + [(self.actions['route_' + str(n_agt)] if self.actions is not None else 1) for n_agt in not_agent_id]
                    + [len(self.arrived_vehicles) / self.n_cars]
                    + [(self.n_cars - len(self.arrived_vehicles)) / self.n_cars]
            )
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
            self.roadPrices[agent_id] = self.pricing_dict[action](
                self.roadPrices[agent_id]
            )
            if self.roadPrices[agent_id] > self.agent_maxes[agent_id]:
                self.agent_maxes[agent_id] = self.roadPrices[agent_id]
                self.agent_price_range[agent_id] = (
                    self.agent_maxes[agent_id] - self.agent_mins[agent_id]
                )
            if self.roadPrices[agent_id] < self.agent_mins[agent_id]:
                self.agent_mins[agent_id] = self.roadPrices[agent_id]
                self.agent_price_range[agent_id] = (
                    self.agent_maxes[agent_id] - self.agent_mins[agent_id]
                )

            self.agent_prices[agent_id] = self.agent_prices[agent_id] + [self.roadPrices[agent_id]]


        # Here, we need to update the simulation and push forward with one timestep
        # Once we've updated all of that, we then update the rewards
        # first, we update the travel time, the queues and the arrived vehicles
        # (
        #     self.roadTravelTime,
        #     self.roadQueues,
        #     self.arrived_vehicles,
        #     self.queues_manager,
        #     self.agent_vdf_cache,
        # ) = quick_get_new_travel_times(
        #     self.roadQueues,
        #     self.roadVDFS,
        #     self.time,
        #     self.arrived_vehicles,
        #     self.time_out_car,
        #     self.queues_manager,
        #     self.agent_vdf_cache,
        # )

        self.roadTravelTime, self.time_out_pq, self.arrived_vehicles = pq_get_new_travel_times(
            self.time_out_pq,
            self.roadVDFS,
            self.time,
            self.arrived_vehicles,
        )
        self.roadQueues = {r: [i[1] for i in list(pq.queue)] for r, pq in self.time_out_pq.items()}
        # print(self.time, self.roadTravelTime, {r: len(x) for r, x in self.roadQueues.items()})
        # next, we update the car arrivals
        num_vehicles_arrived = self.arrival_timestep_dict[self.time]
        cars_arrived_vot = [
            self.vot_deque.popleft() for _ in range(num_vehicles_arrived)
        ]

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
            vehicle_tuple = (
                    decision,
                    self.time,
                    vot,
                    self.time + self.roadTravelTime[decision],
                    self.roadPrices[decision],
                )
            self.roadQueues[decision] = self.roadQueues[decision] + [
                vehicle_tuple
            ]

            # Uncomment below for maximising profit
            if self.reward_fn == 'MaxProfit':
                timestep_rewards[decision] = timestep_rewards[decision] + self.roadPrices[decision]
            # Uncomment bellow for minimising total combined cost
            if self.reward_fn =='MinCombinedCost':
                timestep_rewards[decision] = timestep_rewards[decision] - ((vot*self.roadTravelTime[decision]) + self.roadPrices[decision])
            if self.reward_fn == 'MinVehTravelTime':
                timestep_rewards[decision] = timestep_rewards[decision] - (self.roadTravelTime[decision])

            # self.time_out_car[decision][
            #     round(self.time + self.roadTravelTime[decision])
            # ] = (
            #     self.time_out_car[decision][
            #         round(self.time + self.roadTravelTime[decision])
            #     ]
            #     + 1
            # )
            self.time_out_pq[decision].put((round(self.time + self.roadTravelTime[decision]), vehicle_tuple))

        # Uncomment below for minimising travel time
        # norm_rewards = {}
        # for agent in self.roadVDFS.keys():
            # self.agent_reward_norms[agent] = self.agent_reward_norms[agent] + [timestep_rewards[agent]]
            # agent_reward_mean = np.mean(self.agent_reward_norms[agent])
            # agent_reward_std = np.std(self.agent_reward_norms[agent])
            # timestep_rewards[agent] = -self.roadTravelTime[agent]
            # agent_reward_mean, agent_reward_var = self.update_rolling_norms(
            #     agent, timestep_rewards[agent]
            # )
            # agent_reward_std = np.sqrt(agent_reward_var)
            # norm_timestep_reward = (timestep_rewards[agent] - agent_reward_mean) / (
            #     1 if agent_reward_std == 0 else agent_reward_std
            # )
            # norm_rewards["route_" + str(agent)] = np.float32(timestep_rewards[agent])
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
        self.actions = actions
        rewards = timestep_rewards
        terminations = {"route_" + str(a): False for a in self.roadVDFS.keys()}
        truncations = {"route_" + str(a): False for a in self.roadVDFS.keys()}
        infos = {agent: {'action_mask':
                             [
                                False if self.roadPrices[self.agent_name_mapping[agent]] <= self.price_lower_bound else True,
                                True,
                                False if self.roadPrices[self.agent_name_mapping[agent]] >= self.price_upper_bound else True
                             ]
                         }
            for agent in self.agents
        }
        self.time += 1
        if self.is_simulation_complete():
            self.agents = []
            terminations = {"route_" + str(a): True for a in self.roadVDFS.keys()}
            for road, queue in self.roadQueues.items():
                self.arrived_vehicles = self.arrived_vehicles + [
                    car for car in self.roadQueues[road]
                ]

            self.travel_time = [(c[3] - c[1]) for c in self.arrived_vehicles]
            # print(np.mean(self.travel_time))
            self.time_cost_burden = [
                ((c[3] - c[1]) * c[2]) for c in self.arrived_vehicles
            ]
            self.combined_cost = [
                ((c[3] - c[1]) * c[2]) + c[4] for c in self.arrived_vehicles
            ]
            self.road_profits = {
                r: sum([x[4] for x in self.arrived_vehicles if x[0] == r])
                for r in self.roadQueues.keys()
            }
            # print("END OF SIM ARRIVED CARS", len(self.arrived_vehicles), "VS", self.n_cars)
        return observations, rewards, terminations, truncations, infos

        # else:
        #     self.state[self.agents[1 - self.agent_name_mapping[agent]]] = NONE
        #     self._clear_rewards()
        # self.agent_selection = self._agent_selector.next()
        # self._accumulate_rewards()

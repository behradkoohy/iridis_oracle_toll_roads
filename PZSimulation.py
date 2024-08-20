import functools
from collections import Counter, defaultdict, deque
from functools import partial

import gymnasium.spaces
import numpy as np
import ray
from pettingzoo import AECEnv, ParallelEnv
from pettingzoo.utils import agent_selector
from pettingzoo.test import api_test
from gymnasium.spaces import Discrete, Box

import numpy.random as nprand
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv, ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.pre_checks.env import check_multiagent_environments
from ray.rllib.utils.test_utils import run_rllib_example_script_experiment, add_rllib_example_script_args
from ray.tune import tune
from ray.tune.registry import get_trainable_cls
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.tune.registry import register_env

from TimeOnlyUtils import volume_delay_function, alternative_get_new_travel_times
from TimestepPriceUtils import quantal_decision

n_cars = 100
timeseed = 0
votseed = 0
NONE = 3

class simulation_env(AECEnv):
    # metadata = {'render.modes': []}
    def __init__(self, n_routes=2, render_mode=None):
        self.num_routes = 2
        self.n_routes = n_routes
        self.agents = ['route_' + str(r) for r in range(self.num_routes)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.render_mode = None
        # self.action_spaces = {agent: Discrete(3) for agent in self.possible_agents}

        # parameters that should mostly stay the same. if they need changing, changing in here should edit all exps
        self.timesteps = 30
        self.beta_dist_alpha = 5
        self.beta_dist_beta = 5
        self.n_cars = n_cars

        self.car_vot_upperbound = 9.5
        self.car_vot_lowerbound = 2.5
        self.bound = 1
        self.pricing_dict = {
            -1: lambda x: x - self.bound,
            0: lambda x: x,
            1: lambda x: x + self.bound,
        }


        self.obs_space = Box(low=np.array([0, 0, 0, 15, 0, 0, 15]), high=np.array([self.n_cars, 200, self.n_cars, 1000000, self.n_cars, 200, 1000000]), dtype=np.float64)
        self.act_space = Discrete(3)
        self.observation_space_d = gymnasium.spaces.Dict(
            {
                a_id : Box(low=np.array([0, 0, 0, 15, 0, 0, 15]), high=np.array([self.n_cars, 200, self.n_cars, 1000000, self.n_cars, 200, 1000000]), dtype=np.float64)
                for a_id in self.possible_agents
            }
        )
        self.action_space_d = gymnasium.spaces.Dict(
            {
                a_id : Discrete(3) for a_id in self.possible_agents
            }
        )
        # self.action_space = [self.act_space for _ in range(n_routes)]
        # self.observation_space = [self.obs_space for _ in range(n_routes)]
        # self.observation_space = Box(low=np.array([0, 0, 1, 15, 0, 0, 15]), high=np.array([self.n_cars, 200, self.n_cars, 200, self.n_cars, 200, 1000000]), dtype=np.float64)



    def generate_car_time_distribution(self):

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
        return car_dist_arrival

    def generate_car_vot_distribution(self):
        nprand.seed(votseed)
        car_vot = nprand.uniform(
            self.car_vot_lowerbound, self.car_vot_upperbound, self.n_cars
        )
        return car_vot

    def is_simulation_complete(self):
        if self.time >= self.timesteps:
            return True
        else:
            return False

    # @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        # return Box(low=np.array([0, 0, 0, 15, 0, 0, 15]), high=np.array([self.n_cars, 200, self.n_cars, 1000000, self.n_cars, 200, 1000000]), dtype=np.float64)
        return self.observation_space_d[agent]

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    # @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_space_d[agent]

    def reset(self, seed=None, options=None):
        # This is the stuff that PettingZoo needs
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: NONE for agent in self.agents}
        self.observations = {agent: np.array([0, 0, 1, 15, 0, 0, 15]) for agent in self.agents}
        self.num_moves = 0
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.car_dist_arrival = self.generate_car_time_distribution()
        self.car_vot_arrival = self.generate_car_vot_distribution()
        self.time = 0
        self.roadQueues = {r: [] for r in range(self.num_routes)}
        self.roadVDFS = {
            0: partial(volume_delay_function, 0.656, 4.8, 15, 20),
            1: partial(volume_delay_function, 0.656, 4.8, 30, 20),
        }
        self.roadTravelTime = {r: self.roadVDFS[r](0) for r in self.roadVDFS.keys()}
        self.arrival_timestep_dict = Counter(self.car_dist_arrival)
        self.roadPrices = {r: 20.0 for r in self.roadVDFS.keys()}

        self.time_out_car = {r: defaultdict(int) for r in self.roadVDFS.keys()}
        self.arrived_vehicles = []
        self.vot_deque = deque(self.car_vot_arrival)
        # print(self.observations)
        return self.observations, self.infos



    def observe(self, agent):
        agent_id = self.agent_name_mapping[agent]
        not_agent_id = [x for x in self.agent_name_mapping.values() if x is not agent_id]
        return np.array(
            [
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
        )

    def close(self):
        pass

    def get_utility(self, travel_time, econom_cost, vot):
        return -((vot * travel_time) + econom_cost)

    def generate_utility_funct(self, travel_time, econom_cost):
        return partial(self.get_utility, travel_time, econom_cost)

    def step(self, action):
        if (self.terminations[self.agent_selection] or self.truncations[self.agent_selection]):
            print("how are we here")
            self._was_dead_step(action)
            return
        agent = self.agent_selection
        agent_id = self.agent_name_mapping[agent]
        self._cumulative_rewards[agent] = 0
        self.state[self.agent_selection] = action
        # update the price of the road
        action = action - 1
        self.roadPrices[agent_id] = self.pricing_dict[action](self.roadPrices[agent_id])

        if self._agent_selector.is_last():
            # Here, we need to update the simulation and push forward with one timestep
            # Once we've updated all of that, we then update the rewards
            # first, we update the travel time, the queues and the arrived vehicles
            self.roadTravelTime, self.roadQueues, self.arrived_vehicles = alternative_get_new_travel_times(
                self.roadQueues, self.roadVDFS, self.time, self.arrived_vehicles, self.time_out_car
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
                c: quantal_decision(r) for c, r in car_utilities.items()
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
                timestep_rewards[decision] = timestep_rewards[decision] + 1
                self.time_out_car[decision][round(self.time + self.roadTravelTime[decision])] = (
                        self.time_out_car[decision][round(self.time + self.roadTravelTime[decision])] + 1
                )

            """
            BIG ERROR: WE ARE LOSING VEHICLES SOMEWHERE
            FIND IT
            """
            # This doesn't work: we might miss some values
            if num_vehicles_arrived > 0:
                timestep_rewards = {a: x*self.roadPrices[a] for a,x in timestep_rewards.items()}

                timestep_rewards = {"route_"+str(a): x/sum(timestep_rewards.values()) for a,x in timestep_rewards.items()}
            else:
                timestep_rewards = {"route_" + str(a): 0 for a in self.roadVDFS.keys()}

            self.rewards = timestep_rewards
            self.truncations = {"route_" + str(a): self.is_simulation_complete() for a in self.roadVDFS.keys()}
            self.time += 1
        else:
            self.state[self.agents[1 - self.agent_name_mapping[agent]]] = NONE
            self._clear_rewards()
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()




if __name__ == "__main__":
    env = simulation_env()
    env.reset(seed=42)
    api_test(env)
#     # for agent in env.agent_iter():
#     #     observation, reward, termination, truncation, info = env.last()
#     #     print(observation.shape, observation.dtype)
#     #     if termination or truncation:
#     #         action = None
#     #     else:
#     #         # this is where you would insert your policy
#     #         action = env.action_space(agent).sample()
#     #     env.step(action)
#     # env.close()

def env_creator(env_config):
    return PettingZooEnv(simulation_env(env_config))
    # return simulation_env(env_config)

"""
if __name__ == "__main__":
    parser = add_rllib_example_script_args(
        default_iters=200,
        default_timesteps=1000000,
        default_reward=0.0,
    )
    args = parser.parse_args()
    args.enable_new_api_stack = True
    args.num_agents = 2

    # env = simulation_env()
    register_env("simenv", lambda config: ParallelPettingZooEnv(env_creator(config)))
    env = env_creator({})
    policies = {f"route_{i}" for i in range(2)}
    base_config = (
        get_trainable_cls(args.algo)
        .get_default_config()
        .environment("simenv")
        .multi_agent(
            policies=set(env.env.agents),
            # Exact 1:1 mapping from AgentID to ModuleID.
            policy_mapping_fn=(lambda aid, *args, **kwargs: aid),
        )
        .training(
            vf_loss_coeff=0.005,
        )
        .rl_module(
            model_config_dict={"vf_share_layers": True},
            rl_module_spec=MultiAgentRLModuleSpec(
                module_specs={p: SingleAgentRLModuleSpec() for p in policies},
            ),
        )
    )

    run_rllib_example_script_experiment(base_config, args)

"""
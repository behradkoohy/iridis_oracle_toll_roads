import functools
from collections import Counter
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Pool
from os import times
from random import randint, choices
from typing import Any

from PIL.ImagePalette import random
from gymnasium.spaces import Discrete, Box
from pettingzoo import ParallelEnv
import numpy as np
from scipy.stats import mielke, norm, uniform
from tqdm import tqdm

from TNTP_Parser import parse_network_file, read_trips_file, build_connectivity, all_pairs_shortest_paths, \
    precompute_routes_for_od_pairs, sample_trips, compute_travel_times, calculate_incoming_flows_per_link, \
    precompute_all_route_indices, compute_route_travel_time_from_cache, compute_route_toll_price_from_cache, \
    compute_route_metrics_from_cache

# Pricing functions import
from pricing import LinearPricing, FixedPricing, UnboundPricing
from queue import PriorityQueue

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)


class TNTPParallelEnv(ParallelEnv):
    """
    A PettingZoo Parallel Environment for a traffic network simulation.

    This environment will:
      - Load network and flow data, and build necessary data structures (e.g., connectivity, link mappings).
      - Initialize the simulation state (global clock, vehicle states, etc.).
      - Allow multiple agents (e.g., vehicles or aggregated entities) to act simultaneously.
      - Provide observations, rewards, dones, and info dictionaries for all agents at each step.
    """

    def __init__(self,
                 timesteps=3600,
                 simulation_time=86400,
                 k_depth=2,
                 random_initial_road_cost=False,
                 tntp_path="/Users/behradkoohy/Development/TransportationNetworks/SiouxFalls/SiouxFalls",
                 lambd=0.9,
                 free_roads=False,
                 pricing_mode="step",
                 pricing_params=None,
                 seed=None
        ):
        """
        Initializes the environment.

        Steps:
          - Load network and flow files using existing parsers.
          - Build connectivity dictionaries, link data arrays, and any precomputed metrics.
          - Initialize simulation state variables (e.g., simulation clock, active vehicles, agent IDs).
          - Define the observation and action spaces.
        """
        # Example (replace with actual data-loading and initialization):
        # self.metadata, self.link_data = parse_network_file("network_file.txt")
        # self.flow_metadata, self.od_matrix = parse_flow_file("flow_file.txt")
        # self.connectivity, self.link_index = build_connectivity(self.link_data)
        # self.all_shortest_paths = all_pairs_shortest_paths(self.connectivity)
        # self.agents = [...]  # List of agent IDs (e.g., vehicle IDs)
        # self.current_time = 0
        # self.observation_space = ...  # Define according to your simulation
        # self.action_space = ...       # Define according to your simulation
        self.render_mode = None
        self.timesteps = timesteps
        self.random_initial_road_cost = random_initial_road_cost
        self.free_roads = free_roads

        self.car_vot_upperbound = 0.999
        self.car_vot_lowerbound = 0.001

        self.lambd = lambd

        self.bound = 1
        self.price_lower_bound = self.bound
        self.price_upper_bound = 125

        self.pricing_mode = pricing_mode
        self.pricing_params = pricing_params or []

        # define the paths for the trip and network files
        net_path = tntp_path + "_net.tntp"
        trips_path = tntp_path + "_trips.tntp"

        self.network_metadata, self.link_data = parse_network_file(net_path)
        # network metadata contents: {'NUMBER OF ZONES': 24, 'NUMBER OF NODES': 24, 'FIRST THRU NODE': 1, 'NUMBER OF LINKS': 76, 'ORIGINAL HEADER': '~'}
        # link data contents: a numpy array of shape (n_nodes, 10)
        # where the 10 contents are [init_node, term_node, capacity, length, free_flow_time, B, power, speed_limit, toll, link_type]

        self.trip_metadata, self.od_matrix = read_trips_file(trips_path)
        # trip metadata contents: {'NUMBER OF ZONES': 24, 'TOTAL OD FLOW': 360600.0}
        # od_matrix contents: a dictionary of dictionarys which is the flow amount from origin to demand.

        self.possible_agents = ["link_" + str(s) for s in range(self.network_metadata['NUMBER OF LINKS'])]
        self.agent_name_mapping = dict(
            zip(
                self.possible_agents,
                range(self.network_metadata['NUMBER OF LINKS'])
            )
        )
        self.num_links = len(self.possible_agents)

        # instantiate the appropriate PricingStrategy
        if pricing_mode == "linear":
            self.pricing_strategy = LinearPricing()
        elif pricing_mode == "fixed":
            self.pricing_strategy = FixedPricing()
        elif pricing_mode == "unbound":
            self.pricing_strategy = UnboundPricing()
        else:
            self.pricing_strategy = None
        # reset strategy with parameters if present
        if self.pricing_strategy is not None:
            self.pricing_strategy.reset(self.pricing_params, self)

        # building a connectivity and index mapping dict
        self.connectivity, self.link_index = build_connectivity(self.link_data)
        self.rev_link_index = {v: k for k, v in self.link_index.items()}
        # building a priority queue for the trips which will be used to manage entries and arrivals
        # self.trip_queue = PriorityQueue()

        # this is the default values, stored as numpy arrays so we can vectorise calculations
        self.free_flow_travel_times = self.link_data[:, 4]
        self.B_array = self.link_data[:, 5]
        self.capacity_array = self.link_data[:, 2]
        self.power_array = self.link_data[:, 6]

        # we need to calculate the shortest number of links from one node to another
        self.shortest_paths = all_pairs_shortest_paths(self.connectivity)
        self.shortest_paths = {k: len(v) for k, v in self.shortest_paths.items()}


        # precompute the routes with a depth of min + k, use this to offer the routes to the players
        self.routes_to_offer = {}
        self.route_travel_times = {}
        self.route_toll_cost = {}
        # THIS IS OLD CODE - WE NOW HAVE TO CALCULATE THE TRAVEL TIME AND THE TOLL FOR EACH ARRIVAL
        # NOT JUST ONCE WHEN THE AGENT ENTERS THE SIMULATION. THIS MIGHT BE DIFFICULT.
        for od_pair, shortest_path in self.shortest_paths.items():
            od_pair = (int(od_pair[0]), int(od_pair[1]))
            if od_pair[0] == od_pair[1]:
                continue
            paths = precompute_routes_for_od_pairs(
                [od_pair],
                self.connectivity,
                self.link_index,
                self.link_data,
                shortest_path + k_depth,
            )
            # paths[od_pair] = [r[0] for r in paths[od_pair]]
            self.routes_to_offer[od_pair] = paths[od_pair]

        self.routes_to_index = []
        for od, route in self.routes_to_offer.items():
            for r in route:
                self.routes_to_index.append(r)

        self.routes_to_index.sort(key=lambda x: len(x))

        self.route_indices = precompute_all_route_indices(self.routes_to_index, self.link_index)

        # Pre-bucket routes by length for fast vectorized updates
        self.route_buckets = {}
        for route, idx in self.route_indices.items():
            L = len(idx)
            self.route_buckets.setdefault(L, []).append((route, idx))

        # print({x: len(y) for x, y in self.routes_to_offer.items()})
        self.reset(seed=seed)

    def sample_trips(self, max_time, vot_dist='dagum', timeseed=None, votseed=None):
        """
        Dagum distribution parameters:
        â = 22020.6, b = 2.7926, and c = 0.2977
        """
        def normalise_dist(np_arr):
            return (np_arr - np_arr.min()) / (np_arr.max() - np_arr.min())

        self.seeded_dist = np.random.default_rng(votseed)

        # norm.random_state = seeded_dist
        # Ensure the seed is set for reproducibility
        supported_vot_dists = {
            'normal': lambda x: normalise_dist(norm.rvs(size=x, random_state=self.seeded_dist)),
            'dagum': lambda x: normalise_dist(mielke.rvs(22020.6, 2.7926, size=x, random_state=self.seeded_dist)),
            'uniform': lambda x: normalise_dist(uniform.rvs(size=x, random_state=self.seeded_dist))
        }


        # supported_vot_dists = {
        #     'normal': lambda x: normalise_dist(norm.rvs(size=x, random_state=votseed)),
        #     'dagum': lambda x: normalise_dist(mielke.rvs(22020.6, 2.7926, size=x, random_state=votseed)),
        #     'uniform': lambda x: normalise_dist(uniform.rvs(size=x, random_state=votseed))
        # }


        assert vot_dist in supported_vot_dists.keys(), "vot_dist must be one of {}".format(supported_vot_dists.keys())

        # if timeseed:
        #     np.random.seed(timeseed)
        trips = sample_trips(self.od_matrix, random_state=None if timeseed is None else np.random.RandomState(timeseed))
        trips = [trip for trip in trips if trip['entry_time'] <= max_time]

        # if timeseed:
        #     np.random.seed(timeseed)

        # Set up the player dictionaries here
        player_vots = supported_vot_dists[vot_dist](len(trips))
        for trip, vot in zip(trips, player_vots):
            trip['vot'] = vot
            trip['current_location'] = None
            trip['visited'] = None
            trip['toll_paid'] = 0
            trip['initial_expected_toll'] = None
            trip['entry_time'] = trip['entry_time']

        return trips

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(3)

    def observation_space(self, agent):
        return Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, -1, 0]),
            high=np.array(
                [
                    np.inf,
                    np.inf,
                    self.n_cars,
                    150,
                    self.n_cars,
                    self.n_cars,
                    np.inf,
                    2,
                    self.n_cars
                ]
            ),
            dtype=np.float64,
        )

    def reset(self, seed=None, free_roads=False):
        """
        Resets the environment to its initial state.

        Steps:
          - Reset simulation clock and network state.
          - Clear active vehicle states and reinitialize the agent list.
          - Return an initial observation dictionary mapping each agent to its observation.
        """
        # Reset simulation state (e.g., self.current_time = 0, self.vehicles = initial_vehicle_list)
        # Generate and return a dict of initial observations for each agent.
        # Example:
        # observations = {agent: self._get_initial_observation(agent) for agent in self.agents}
        # return observations
        self.trips = self.sample_trips(self.timesteps, timeseed=seed, votseed=seed)
        self.completed_players = []
        self.free_roads = free_roads

        # precompute the number of players that arrive at each timestep.
        self.arrival_spread = Counter([trip['entry_time'] for trip in self.sample_trips(self.timesteps, timeseed=seed, votseed=seed)])

        # self.trip_departure_times = {}
        # for x in range(self.timesteps):
        self.agents = self.possible_agents[:]
        self.num_moves = 0

        self.flows = np.zeros(self.free_flow_travel_times.shape)
        self.n_cars = len(self.trips)
        if not self.free_roads:
            self.tolls = np.array([20 if not self.random_initial_road_cost else randint(1,40) for _ in list(self.link_index.keys())])
        else:
            self.tolls = np.array([0 for _ in list(self.link_index.keys())])

        self.travel_times_vectorized = compute_travel_times(
            self.flows,
            self.free_flow_travel_times,
            self.B_array,
            self.capacity_array,
            self.power_array
        )

        # for od_pair, routes in self.routes_to_offer.items():
        #     for route in routes:
        #         route_travel_time = compute_route_travel_time(route, self.link_index, self.travel_times_vectorized)
        #         self.route_travel_times[route] = route_travel_time
        self.update_route_time_and_price()

        self.actions = None


        """
        Instead of creating a dictionary of every player and their arrival time, why don't i make a heap or priority queue?
        I need to track which nodes they've already been to, but then I could also add players that are mid simulation 
        into the same queue
        
        how do we track which players are on which road? we'd use the 'flow' matrix or whatever it's called
        this would be the best way to play it.
        """
        self.time = 0
        # building a priority queue for the trips which will be used to manage entries and arrivals
        self.trip_queue = PriorityQueue()
        for player in self.trips:
            player['current_location'] = player['origin']
            player['visited'] = [player['origin']]
            player['time_travelled'] = 0
            self.trip_queue.put(PrioritizedItem(player['entry_time'], player))

        self.incoming_flows = calculate_incoming_flows_per_link(self.link_data, self.flows)
        observations = {agent: self.get_observe(agent) for agent in self.agents}
        infos = {agent: {'action_mask':
            [
                False if self.tolls[self.agent_name_mapping[agent]] <= self.price_lower_bound else True,
                True,
                False if self.tolls[self.agent_name_mapping[agent]] >= self.price_upper_bound else True
            ]
        }
            for agent in self.agents
        }
        return observations, infos

    def get_observe(self, agent):
        # agent_id = self.link_index[agent]
        agent_id = self.agent_name_mapping[agent]
        agent_ff_tt = self.link_data[:, 4][agent_id]
        agent_capacity = self.link_data[:, 2][agent_id]
        players_on_road = self.flows[agent_id]
        tolls = self.tolls[agent_id]
        agent_start = self.rev_link_index[agent_id][0]
        incoming_flows = self.incoming_flows[agent_start]

        number_of_vehicles_in_sim = self.trip_queue.qsize()
        curr_travel_time = self.travel_times_vectorized[agent_id]
        previous_action = self.actions["link_" + str(agent_id)] if self.actions is not None else 1
        timestep_entries = self.arrival_spread[self.timesteps]
        return np.array(
                [
                agent_ff_tt,
                agent_capacity,
                players_on_road,
                tolls,
                incoming_flows,
                number_of_vehicles_in_sim,
                curr_travel_time,
                previous_action,
                timestep_entries
            ]
        )

    def get_arrivals_for_current_timestep(self):
        if self.trip_queue.empty():
            return []
        arrivals = []
        top_item = self.trip_queue.get()
        if top_item.priority == self.time:
            arrivals.append(top_item.item)
            while not self.trip_queue.empty():
                top_item = self.trip_queue.get()
                if top_item.priority == self.time:
                    arrivals.append(top_item.item)
                else:
                    self.trip_queue.put(top_item)
                    break
        else:
            self.trip_queue.put(top_item)

        return arrivals


    # def update_route_times(self):
    #     for od_pair, routes in self.routes_to_offer.items():
    #         for route in routes:
    #             route_travel_time = compute_route_travel_time(route, self.link_index, self.travel_times_vectorized)
    #             self.route_travel_times[tuple(route)] = route_travel_time

    # def route_update_funct(self, route):
    #     route_travel_time = compute_route_travel_time_from_cache(route, self.route_indices,
    #                                                              self.travel_times_vectorized)
    #     route_toll = compute_route_toll_price_from_cache(route, self.route_indices, self.tolls)
    #     return tuple((tuple(route), (route_travel_time, route_toll)))

    # def _apply_route_update_funct(self, route):
    #     route_travel_time, route_toll = compute_route_metrics_from_cache(
    #         route,
    #         self.route_indices,
    #         self.travel_times_vectorized,
    #         self.tolls
    #     )
    #     return (tuple(route), route_travel_time), (tuple(route), route_toll)

    # def update_route_time_and_price(self):
    #     for route in self.routes_to_index:
    #         route_travel_time, route_toll = compute_route_metrics_from_cache(
    #             route,
    #             self.route_indices,
    #             self.travel_times_vectorized,
    #             self.tolls
    #         )
    #         self.route_travel_times[tuple(route)] = route_travel_time
    #         self.route_toll_cost[tuple(route)] = route_toll

    def update_route_time_and_price(self):
        """
        Vectorized update of travel times and tolls for all routes, grouped by route length.
        """
        new_travel_times = {}
        new_route_tolls = {}
        # For each group of routes with the same number of links
        for L, bucket in self.route_buckets.items():
            # Build a (N, L) index matrix
            idx_mat = np.stack([idx for (_route, idx) in bucket], axis=0)
            # Sum travel times and tolls along each route
            times = self.travel_times_vectorized[idx_mat].sum(axis=1)
            tolls = self.tolls[idx_mat].sum(axis=1)
            # Write back into dicts
            for i, (route, _) in enumerate(bucket):
                new_travel_times[route] = times[i]
                new_route_tolls[route]   = tolls[i]
        # Atomically replace the stored values
        self.route_travel_times = new_travel_times
        self.route_toll_cost    = new_route_tolls





    def is_simulation_complete(self):
        if self.trip_queue.empty():
            return True
        else:
            if self.time > self.timesteps:
                pass
            else:
                pass
        return False
        # if self.time > self.timesteps:
        #     return True
        # else:
        #     return False

    def step(self, actions):
        """
        Executes one simulation step for all agents simultaneously.

        Parameters:
          actions: A dictionary mapping agent IDs to their chosen actions.

        Steps:
          - Process all agents' actions concurrently.
          - Update the simulation state (e.g., vehicle positions, link flows, simulation clock).
          - Compute rewards for each agent.
          - Determine which agents are done and collect additional info.
          - Return updated observations, rewards, dones, and info dictionaries.

        Returns:
          observations: dict mapping agent IDs to new observations.
          rewards: dict mapping agent IDs to reward values.
          dones: dict mapping agent IDs to boolean done flags.
          infos: dict mapping agent IDs to additional information.
        """

        # print(self.time)

        self.actions = actions
        # e.g. {'route_0': np.int64(1), 'route_1': np.int64(0), 'route_2': np.int64(0)}
        # update road prices
        if not self.free_roads:
            # step-mode discrete adjustments
            if self.pricing_mode == "step" or self.pricing_strategy is None:
                values = np.array(list(self.actions.values()), dtype=int)
                deltas = values - 1
                self.tolls = self.tolls + deltas
            else:
                # delegate to instantiated PricingStrategy
                self.tolls = self.pricing_strategy.get_tolls(self, self.time)

            # clip to bounds
            self.tolls = np.clip(self.tolls,
                                 self.price_lower_bound,
                                 self.price_upper_bound)
        else:
            self.tolls = np.zeros_like(self.tolls)
        # calculate new road travel times
        self.travel_times_vectorized = compute_travel_times(
            self.flows,
            self.free_flow_travel_times,
            self.B_array,
            self.capacity_array,
            self.power_array
        )

        self.update_route_time_and_price()
        players_to_allocate = self.get_arrivals_for_current_timestep()

        agent_rewards = {agt: 0 for agt in self.link_index.keys()}

        for player in players_to_allocate:
            if player['current_location'] == player['destination']:
                self.completed_players.append(player)
                continue
            else:
                # NOTE: this is where it starts to get a bit funky.
                # NOTE: we're precomputing the routes with a flow of 0. this may mean that a quicker route exists
                # NOTE: in the instance that it does, we won't know bc it's not considered.
                # NOTE: future releases will fix this bug.
                player_routes = self.routes_to_offer[(player['current_location'], player['destination'])]
                player_routes = [route for route in player_routes]
                timed_routes = np.array([self.route_travel_times[tuple(route)] for route in player_routes])
                costed_routes = np.array([self.route_toll_cost[tuple(route)] for route in player_routes])
                route_utility = - ((timed_routes * player['vot']) + costed_routes)

                # we need to calculate the quantal response funct so we're going to use a variation of the softmax fn
                route_utility = np.exp(route_utility - np.max(route_utility))
                route_utility = route_utility / np.sum(route_utility)
                # breakpoint()
                player_routes_enum = [(i, x) for i, x in enumerate(route_utility)]
                route_choice = self.seeded_dist.choice([p[0] for p in player_routes_enum], size=1, p=[p[1] for p in player_routes_enum])
                route_choice = player_routes[route_choice[0]]
                next_location = route_choice[1]  # Get the next location from the chosen route

                # route_choice = choices(player_routes, weights=route_utility)
                # next_location = route_choice[0][1]
                # print(route_choice, next_location)

                time_to_travel = self.route_travel_times[(player['current_location'], next_location)]
                cost_to_travel = self.route_toll_cost[(player['current_location'], next_location)]
                player['toll_paid'] += cost_to_travel
                player['time_travelled'] += time_to_travel

                agent_rewards[(int(player['current_location']), int(next_location))] = agent_rewards[(int(player['current_location']), int(next_location))] + cost_to_travel
                self.trip_queue.put(PrioritizedItem(self.time + round(time_to_travel), player))

                player['current_location'] = next_location
                player['visited'].append(next_location)

                """
                This is the really confusing part
                
                we have a list of routes to offer but some of them will be the same for this segment of the journey
                if we check all the routes then cut it down, we're doing needless computation and we still need to
                reduce the routes to the ones for the current timestep?
                
                if we have multiple routes which all have the same first step, we take the minimum right?
                how do we even go about something like that without going into O(N^2) or worse?
                
                OK - we offer all the routes to the user and then just make the first jump. I think thats the
                most inconsequential way of doing it.
                
                i really wish pycharm would stop giving me suggestions whilst i'm typing
                """
        # NEVER GET OBSERVATIONS WITHOUT RUNNING THE CALCULATE INCOMING FLOWS
        self.incoming_flows = calculate_incoming_flows_per_link(self.link_data, self.flows)
        observations = {agent: self.get_observe(agent) for agent in self.agents}
        rewards = {self.link_index[agt]: reward for agt, reward in agent_rewards.items()}
        terminations = {agt: False for agt in self.agents}
        truncations = {agt: False for agt in self.agents}
        infos = {agent: {'action_mask':
            [
                False if self.tolls[self.agent_name_mapping[agent]] <= self.price_lower_bound else True,
                True,
                False if self.tolls[self.agent_name_mapping[agent]] >= self.price_upper_bound else True
            ]
        }
            for agent in self.agents
        }
        self.time += 1
        if self.is_simulation_complete():
            self.agents = []
            terminations = {a: True for a in self.possible_agents}
            # DO EVAL HERE
            self.travel_time = [player['time_travelled'] for player in self.completed_players]
            self.time_cost_burden = [player['time_travelled']*player['vot'] for player in self.completed_players]
            self.combined_cost = [(player['time_travelled']*player['vot']) + player['toll_paid'] for player in self.completed_players]

        return observations, rewards, terminations, truncations, infos

    def render(self, mode="human"):
        """
        Renders the current state of the simulation.

        For 'human' mode, you might visualize the network state using matplotlib or another library.
        Other modes might return an image array or a textual summary.
        """
        # Render the current state (e.g., network visualization, vehicle positions).
        pass

    def close(self):
        """
        Performs cleanup operations such as closing windows or freeing resources.
        """
        # Clean up any allocated resources.
        pass

    # Additional helper functions can be defined below:
    # e.g., _get_observation(agent), _compute_reward(agent), _update_simulation(), etc.

if __name__ == "__main__":
    env = TNTPParallelEnv(timesteps=3600)
    with tqdm(total=env.timesteps) as pbar:
        while env.agents:
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            observations, rewards, terminations, truncations, infos = env.step(actions)
            pbar.update(1)
    # while env.agents:
    #     # this is where you would insert your policy
    #     actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    #
    #     observations, rewards, terminations, truncations, infos = env.step(actions)

from collections import Counter, defaultdict, deque
from functools import partial
from random import choices

from TimeOnlyUtils import (
    volume_delay_function,
    reduced_is_simulation_complete,
    alternative_get_new_travel_times,
)
import numpy as np
from numpy import mean, quantile, median, std
from numpy import min as nmin
from numpy import max as nmax
import inequalipy as ineq


def quantal_decision(routes):
    # We pass in a list of 2-tuple - (road, utility) for each road.
    utility = [u[1] for u in routes]
    utility = [u - max(utility) for u in utility]
    quantal_weights = shortform_quantal_function(utility)
    choice = choices(routes, weights=quantal_weights)
    # print("101:", [q/sum(quantal_weights) for q in quantal_weights], utility)
    return choice[0]


def shortform_quantal_function(utilities, lambd=0.5):
    return [1 / (1 + np.exp(lambd * ((2 * u) - sum(utilities)))) for u in utilities]


def get_utility(travel_time, econom_cost, vot):
    return -((vot * travel_time) + econom_cost)


def generate_utility_funct(travel_time, econom_cost):
    return partial(get_utility, travel_time, econom_cost)


def evaluate_solution_timestepprice(
    solution,
    car_dist_arrival,
    car_vot_dist,
    pricing_dict,
    timesteps,
    post_eval=False,
    seq_decisions=False,
    optimise_for="TravelTime",
):
    """
    solution: list of roads for the vehicles to take, i.e. [1,1,2,2,1,1,2,....,]
    car_dist_arrival: list of arrival times of vehicles, length n, i.e. [1,1,2,3,...,30]
    """

    roadQueues = {r: [] for r in [1, 2]}
    roadVDFS = {
        1: partial(volume_delay_function, 0.656, 4.8, 15, 20),
        2: partial(volume_delay_function, 0.656, 4.8, 30, 20),
    }
    roadTravelTime = {r: roadVDFS[r](0) for r in roadVDFS.keys()}
    arrival_timestep_dict = Counter(car_dist_arrival)
    arrived_vehicles = []
    roadPrices = {r: 20.0 for r in roadVDFS.keys()}

    time_out_car = {r: defaultdict(int) for r in roadVDFS.keys()}

    arrived_vehicles = []
    time = 0

    sol_deque = deque(solution)
    car_vot_deque = deque(car_vot_dist)
    while not reduced_is_simulation_complete(roadQueues, time, timesteps):
        # get the new vehicles at this timestep

        roadTravelTime, roadQueues, arrived_vehicles = alternative_get_new_travel_times(
            roadQueues, roadVDFS, time, arrived_vehicles, time_out_car
        )
        roadPrices = {
            r: pricing_dict[sol_deque.popleft()](roadPrices[r]) for r in roadVDFS.keys()
        }
        roadPrices = {r: 1 if x < 1 else x for r, x in roadPrices.items()}
        # Add new vehicles from here
        num_vehicles_arrived = arrival_timestep_dict[time]

        cars_arrived = [car_vot_deque.popleft() for _ in range(num_vehicles_arrived)]
        # We need to change this so cars make the decision quantally, and we adjust the pricing instead
        road_partial_funct = {
            r: generate_utility_funct(roadTravelTime[r], roadPrices[r])
            for r in roadVDFS.keys()
        }
        car_utilities = {
            n: list(
                {
                    r: utility_funct(n_car_vot)
                    for r, utility_funct in road_partial_funct.items()
                }.items()
            )
            for n, n_car_vot in enumerate(cars_arrived)
        }
        car_quantal_decision = {
            c: quantal_decision(r) for c, r in car_utilities.items()
        }
        decisions = zip([d[0] for d in car_quantal_decision.values()], cars_arrived)

        """
        Here you need to generate the list of utilities for each vehicle/road combo
        and then put them into a decision list so they can be allocated to the correct road.
        """
        # add vehicles to the new queue
        for decision, vot in decisions:
            roadQueues[decision] = roadQueues[decision] + [
                (
                    decision,
                    time,
                    vot,
                    time + roadTravelTime[decision],
                    roadPrices[decision],
                )
            ]
            time_out_car[decision][round(time + roadTravelTime[decision])] = (
                time_out_car[decision][round(time + roadTravelTime[decision])] + 1
            )
            if seq_decisions:
                roadTravelTime, roadQueues, arrived_vehicles = (
                    alternative_get_new_travel_times(
                        roadQueues, roadVDFS, time, arrived_vehicles, time_out_car
                    )
                )
        time += 1
    for road, queue in roadQueues.items():
        arrived_vehicles = arrived_vehicles + [car for car in roadQueues[road]]
    travel_time = [c[3] - c[1] for c in arrived_vehicles]
    time_cost_burden = [(c[3] - c[1]) * c[2] for c in arrived_vehicles]
    combined_cost = [((c[3] - c[1]) * c[2]) + c[4] for c in arrived_vehicles]
    if post_eval:
        return (
            (
                nmin(time_cost_burden),
                quantile(time_cost_burden, 0.25),
                mean(time_cost_burden),
                median(time_cost_burden),
                quantile(time_cost_burden, 0.75),
                nmax(time_cost_burden),
                std(time_cost_burden),
                ineq.gini(time_cost_burden),
                ineq.atkinson.index(time_cost_burden, epsilon=0.5),
            ),
            (
                nmin(travel_time),
                quantile(travel_time, 0.25),
                mean(travel_time),
                median(travel_time),
                quantile(travel_time, 0.75),
                nmax(travel_time),
                std(travel_time),
                ineq.gini(travel_time),
                ineq.atkinson.index(travel_time, epsilon=0.5),
            ),
            (
                nmin(combined_cost),
                quantile(combined_cost, 0.25),
                mean(combined_cost),
                median(combined_cost),
                quantile(combined_cost, 0.75),
                nmax(combined_cost),
                std(combined_cost),
                ineq.gini(combined_cost),
                ineq.atkinson.index(combined_cost, epsilon=0.5),
            ),
        )
    return {
        "TravelTime": -mean(travel_time),
        "SocialCost": -mean(time_cost_burden),
        "CombinedCost": -mean(combined_cost),
    }[optimise_for]


def evaluate_solution_unboundprice(
    solution,
    car_dist_arrival,
    car_vot_dist,
    timesteps,
    post_eval=False,
    seq_decisions=False,
    optimise_for="TravelTime",
):
    """
    solution: list of roads for the vehicles to take, i.e. [1,1,2,2,1,1,2,....,]
    car_dist_arrival: list of arrival times of vehicles, length n, i.e. [1,1,2,3,...,30]
    """

    roadQueues = {r: [] for r in [1, 2]}
    roadVDFS = {
        1: partial(volume_delay_function, 0.656, 4.8, 15, 20),
        2: partial(volume_delay_function, 0.656, 4.8, 30, 20),
    }
    roadTravelTime = {r: roadVDFS[r](0) for r in roadVDFS.keys()}
    arrival_timestep_dict = Counter(car_dist_arrival)
    arrived_vehicles = []
    roadPrices = {r: 20.0 for r in roadVDFS.keys()}

    time_out_car = {r: defaultdict(int) for r in roadVDFS.keys()}

    arrived_vehicles = []
    time = 0

    sol_deque = deque(solution)
    car_vot_deque = deque(car_vot_dist)
    while not reduced_is_simulation_complete(roadQueues, time, timesteps):
        # get the new vehicles at this timestep

        roadTravelTime, roadQueues, arrived_vehicles = alternative_get_new_travel_times(
            roadQueues, roadVDFS, time, arrived_vehicles, time_out_car
        )
        roadPrices = {r: sol_deque.popleft() for r in roadVDFS.keys()}
        roadPrices = {r: 1 if x < 1 else x for r, x in roadPrices.items()}
        # Add new vehicles from here
        num_vehicles_arrived = arrival_timestep_dict[time]

        cars_arrived = [car_vot_deque.popleft() for _ in range(num_vehicles_arrived)]
        # We need to change this so cars make the decision quantally, and we adjust the pricing instead
        road_partial_funct = {
            r: generate_utility_funct(roadTravelTime[r], roadPrices[r])
            for r in roadVDFS.keys()
        }
        car_utilities = {
            n: list(
                {
                    r: utility_funct(n_car_vot)
                    for r, utility_funct in road_partial_funct.items()
                }.items()
            )
            for n, n_car_vot in enumerate(cars_arrived)
        }
        car_quantal_decision = {
            c: quantal_decision(r) for c, r in car_utilities.items()
        }
        decisions = zip([d[0] for d in car_quantal_decision.values()], cars_arrived)

        """
        Here you need to generate the list of utilities for each vehicle/road combo
        and then put them into a decision list so they can be allocated to the correct road.
        """
        # add vehicles to the new queue
        for decision, vot in decisions:
            roadQueues[decision] = roadQueues[decision] + [
                (
                    decision,
                    time,
                    vot,
                    time + roadTravelTime[decision],
                    roadPrices[decision],
                )
            ]
            time_out_car[decision][round(time + roadTravelTime[decision])] = (
                time_out_car[decision][round(time + roadTravelTime[decision])] + 1
            )
            if seq_decisions:
                roadTravelTime, roadQueues, arrived_vehicles = (
                    alternative_get_new_travel_times(
                        roadQueues, roadVDFS, time, arrived_vehicles, time_out_car
                    )
                )
        time += 1
    for road, queue in roadQueues.items():
        arrived_vehicles = arrived_vehicles + [car for car in roadQueues[road]]
    travel_time = [c[3] - c[1] for c in arrived_vehicles]
    time_cost_burden = [(c[3] - c[1]) * c[2] for c in arrived_vehicles]
    combined_cost = [((c[3] - c[1]) * c[2]) + c[4] for c in arrived_vehicles]
    if post_eval:
        return (
            (
                nmin(time_cost_burden),
                quantile(time_cost_burden, 0.25),
                mean(time_cost_burden),
                median(time_cost_burden),
                quantile(time_cost_burden, 0.75),
                nmax(time_cost_burden),
                std(time_cost_burden),
                ineq.gini(time_cost_burden),
                ineq.atkinson.index(time_cost_burden, epsilon=0.5),
            ),
            (
                nmin(travel_time),
                quantile(travel_time, 0.25),
                mean(travel_time),
                median(travel_time),
                quantile(travel_time, 0.75),
                nmax(travel_time),
                std(travel_time),
                ineq.gini(travel_time),
                ineq.atkinson.index(travel_time, epsilon=0.5),
            ),
            (
                nmin(combined_cost),
                quantile(combined_cost, 0.25),
                mean(combined_cost),
                median(combined_cost),
                quantile(combined_cost, 0.75),
                nmax(combined_cost),
                std(combined_cost),
                ineq.gini(combined_cost),
                ineq.atkinson.index(combined_cost, epsilon=0.5),
            ),
        )
    return {
        "TravelTime": -mean(travel_time),
        "SocialCost": -mean(time_cost_burden),
        "CombinedCost": -mean(combined_cost),
    }[optimise_for]


def evaluate_solution_fixedprice(
    pricing_dict,
    car_dist_arrival,
    car_vot_dist,
    timesteps,
    post_eval=False,
    seq_decisions=False,
    optimise_for="TravelTime",
):
    """
    solution: list of roads for the vehicles to take, i.e. [1,1,2,2,1,1,2,....,]
    car_dist_arrival: list of arrival times of vehicles, length n, i.e. [1,1,2,3,...,30]
    """

    roadQueues = {r: [] for r in [1, 2]}
    roadVDFS = {
        1: partial(volume_delay_function, 0.656, 4.8, 15, 20),
        2: partial(volume_delay_function, 0.656, 4.8, 30, 20),
    }
    roadTravelTime = {r: roadVDFS[r](0) for r in roadVDFS.keys()}
    arrival_timestep_dict = Counter(car_dist_arrival)
    arrived_vehicles = []
    roadPrices = pricing_dict

    time_out_car = {r: defaultdict(int) for r in roadVDFS.keys()}

    arrived_vehicles = []
    time = 0

    car_vot_deque = deque(car_vot_dist)
    while not reduced_is_simulation_complete(roadQueues, time, timesteps):
        # get the new vehicles at this timestep

        roadTravelTime, roadQueues, arrived_vehicles = alternative_get_new_travel_times(
            roadQueues, roadVDFS, time, arrived_vehicles, time_out_car
        )
        # Add new vehicles from here
        num_vehicles_arrived = arrival_timestep_dict[time]

        cars_arrived = [car_vot_deque.popleft() for _ in range(num_vehicles_arrived)]
        # We need to change this so cars make the decision quantally, and we adjust the pricing instead
        road_partial_funct = {
            r: generate_utility_funct(roadTravelTime[r], roadPrices[r])
            for r in roadVDFS.keys()
        }
        car_utilities = {
            n: list(
                {
                    r: utility_funct(n_car_vot)
                    for r, utility_funct in road_partial_funct.items()
                }.items()
            )
            for n, n_car_vot in enumerate(cars_arrived)
        }
        car_quantal_decision = {
            c: quantal_decision(r) for c, r in car_utilities.items()
        }
        decisions = zip([d[0] for d in car_quantal_decision.values()], cars_arrived)

        """
        Here you need to generate the list of utilities for each vehicle/road combo
        and then put them into a decision list so they can be allocated to the correct road.
        """
        # add vehicles to the new queue
        for decision, vot in decisions:
            roadQueues[decision] = roadQueues[decision] + [
                (
                    decision,
                    time,
                    vot,
                    time + roadTravelTime[decision],
                    roadPrices[decision],
                )
            ]
            time_out_car[decision][round(time + roadTravelTime[decision])] = (
                time_out_car[decision][round(time + roadTravelTime[decision])] + 1
            )
            if seq_decisions:
                roadTravelTime, roadQueues, arrived_vehicles = (
                    alternative_get_new_travel_times(
                        roadQueues, roadVDFS, time, arrived_vehicles, time_out_car
                    )
                )
        time += 1
    for road, queue in roadQueues.items():
        arrived_vehicles = arrived_vehicles + [car for car in roadQueues[road]]
    travel_time = [c[3] - c[1] for c in arrived_vehicles]
    time_cost_burden = [(c[3] - c[1]) * c[2] for c in arrived_vehicles]
    combined_cost = [((c[3] - c[1]) * c[2]) + c[4] for c in arrived_vehicles]
    if post_eval:
        return (
            (
                nmin(time_cost_burden),
                quantile(time_cost_burden, 0.25),
                mean(time_cost_burden),
                median(time_cost_burden),
                quantile(time_cost_burden, 0.75),
                nmax(time_cost_burden),
                std(time_cost_burden),
                ineq.gini(time_cost_burden),
                ineq.atkinson.index(time_cost_burden, epsilon=0.5),
            ),
            (
                nmin(travel_time),
                quantile(travel_time, 0.25),
                mean(travel_time),
                median(travel_time),
                quantile(travel_time, 0.75),
                nmax(travel_time),
                std(travel_time),
                ineq.gini(travel_time),
                ineq.atkinson.index(travel_time, epsilon=0.5),
            ),
            (
                nmin(combined_cost),
                quantile(combined_cost, 0.25),
                mean(combined_cost),
                median(combined_cost),
                quantile(combined_cost, 0.75),
                nmax(combined_cost),
                std(combined_cost),
                ineq.gini(combined_cost),
                ineq.atkinson.index(combined_cost, epsilon=0.5),
            ),
        )
    return {
        "TravelTime": -mean(travel_time),
        "SocialCost": -mean(time_cost_burden),
        "CombinedCost": -mean(combined_cost),
    }[optimise_for]


def evaluate_solution_linearprice(
    pricing_dict,
    car_dist_arrival,
    car_vot_dist,
    timesteps,
    post_eval=False,
    seq_decisions=False,
    optimise_for="TravelTime",
):
    """
    solution: list of roads for the vehicles to take, i.e. [1,1,2,2,1,1,2,....,]
    car_dist_arrival: list of arrival times of vehicles, length n, i.e. [1,1,2,3,...,30]
    """

    roadQueues = {r: [] for r in [1, 2]}
    roadVDFS = {
        1: partial(volume_delay_function, 0.656, 4.8, 15, 20),
        2: partial(volume_delay_function, 0.656, 4.8, 30, 20),
    }
    roadTravelTime = {r: roadVDFS[r](0) for r in roadVDFS.keys()}
    arrival_timestep_dict = Counter(car_dist_arrival)
    arrived_vehicles = []
    # pricing dict is passed in as {1: (m1, c1), 2: (m2, c2)}
    roadPrices_funct = {
        1: lambda x: (pricing_dict[1][0] * x) + pricing_dict[1][1],
        2: lambda x: (pricing_dict[2][0] * x) + pricing_dict[2][1],
    }
    roadPrices = {r: roadPrices_funct[r](0) for r in roadVDFS.keys()}

    time_out_car = {r: defaultdict(int) for r in roadVDFS.keys()}

    arrived_vehicles = []
    time = 0

    car_vot_deque = deque(car_vot_dist)
    while not reduced_is_simulation_complete(roadQueues, time, timesteps):
        # get the new vehicles at this timestep

        roadTravelTime, roadQueues, arrived_vehicles = alternative_get_new_travel_times(
            roadQueues, roadVDFS, time, arrived_vehicles, time_out_car
        )
        roadPrices = {
            r: roadPrices_funct[r](len(roadQueues[r])) for r in roadVDFS.keys()
        }
        # Add new vehicles from here
        num_vehicles_arrived = arrival_timestep_dict[time]

        cars_arrived = [car_vot_deque.popleft() for _ in range(num_vehicles_arrived)]
        # We need to change this so cars make the decision quantally, and we adjust the pricing instead
        road_partial_funct = {
            r: generate_utility_funct(roadTravelTime[r], roadPrices[r])
            for r in roadVDFS.keys()
        }
        car_utilities = {
            n: list(
                {
                    r: utility_funct(n_car_vot)
                    for r, utility_funct in road_partial_funct.items()
                }.items()
            )
            for n, n_car_vot in enumerate(cars_arrived)
        }
        car_quantal_decision = {
            c: quantal_decision(r) for c, r in car_utilities.items()
        }
        decisions = zip([d[0] for d in car_quantal_decision.values()], cars_arrived)

        """
        Here you need to generate the list of utilities for each vehicle/road combo
        and then put them into a decision list so they can be allocated to the correct road.
        """
        # add vehicles to the new queue
        for decision, vot in decisions:
            roadQueues[decision] = roadQueues[decision] + [
                (
                    decision,
                    time,
                    vot,
                    time + roadTravelTime[decision],
                    roadPrices[decision],
                )
            ]
            time_out_car[decision][round(time + roadTravelTime[decision])] = (
                time_out_car[decision][round(time + roadTravelTime[decision])] + 1
            )
            if seq_decisions:
                roadTravelTime, roadQueues, arrived_vehicles = (
                    alternative_get_new_travel_times(
                        roadQueues, roadVDFS, time, arrived_vehicles, time_out_car
                    )
                )
        time += 1
    for road, queue in roadQueues.items():
        arrived_vehicles = arrived_vehicles + [car for car in roadQueues[road]]
    travel_time = [c[3] - c[1] for c in arrived_vehicles]
    time_cost_burden = [(c[3] - c[1]) * c[2] for c in arrived_vehicles]
    combined_cost = [((c[3] - c[1]) * c[2]) + c[4] for c in arrived_vehicles]
    if post_eval:
        return (
            (
                nmin(time_cost_burden),
                quantile(time_cost_burden, 0.25),
                mean(time_cost_burden),
                median(time_cost_burden),
                quantile(time_cost_burden, 0.75),
                nmax(time_cost_burden),
                std(time_cost_burden),
                ineq.gini(time_cost_burden),
                ineq.atkinson.index(time_cost_burden, epsilon=0.5),
            ),
            (
                nmin(travel_time),
                quantile(travel_time, 0.25),
                mean(travel_time),
                median(travel_time),
                quantile(travel_time, 0.75),
                nmax(travel_time),
                std(travel_time),
                ineq.gini(travel_time),
                ineq.atkinson.index(travel_time, epsilon=0.5),
            ),
            (
                nmin(combined_cost),
                quantile(combined_cost, 0.25),
                mean(combined_cost),
                median(combined_cost),
                quantile(combined_cost, 0.75),
                nmax(combined_cost),
                std(combined_cost),
                ineq.gini(combined_cost),
                ineq.atkinson.index(combined_cost, epsilon=0.5),
            ),
        )

    return {
        "TravelTime": -mean(travel_time),
        "SocialCost": -mean(time_cost_burden),
        "CombinedCost": -mean(combined_cost),
    }[optimise_for]

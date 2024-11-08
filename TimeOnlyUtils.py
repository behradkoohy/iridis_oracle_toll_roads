from collections import defaultdict, Counter, deque
from functools import partial, lru_cache
import numpy as np
from numpy import mean, quantile, median, std
from numpy import min as nmin
from numpy import max as nmax
import inequalipy as ineq



def reduced_is_simulation_complete(roadQueues, time, timesteps):
    if time >= timesteps + 1:
        return True
    else:
        return False

@lru_cache(maxsize=None)
def volume_delay_function(a, b, c, t0, v):
    """
    :param v: volume of cars on road currently
    :param a: VDF calibration parameter alpha, values of 0.15 default
    :param b: VDF calibration parameter beta, value of 4 default
    :param c: road capacity
    :param t0: free flow travel-time
    :return: travel time for a vehicle on road

    link for default value info:
    https://www.degruyter.com/document/doi/10.1515/eng-2022-0022/html?lang=en
    """
    a = 0.656
    b = 4.8
    return t0 * (1 + (a * ((v / c) ** b)))

class QueueRanges:
    def __init__(self, num_routes):
        self.num_routes = num_routes
        # self.starts = {0:0, 1:0}
        # self.stops = {0:0, 1:0}
        # self.queues = {0:{}, 1:{}}
        self.starts = {n: 0 for n in range(self.num_routes)}
        self.stops = {n: 0 for n in range(self.num_routes)}
        self.queues = {n: {} for n in range(self.num_routes)}

    def reset(self):
        self.starts = {n: 0 for n in range(self.num_routes)}
        self.stops = {n: 0 for n in range(self.num_routes)}
        self.queues = {n: {} for n in range(self.num_routes)}

def get_cars_leaving_during_trip(time_out_car, road, time, max_travel_eta):
    road_dict = time_out_car[road]  # Pre-fetch the dictionary for the specific road
    end_time = round(max_travel_eta) + 1  # Calculate range endpoint once
    timesteps_to_check = [
        ti for ti in range(time + 1, end_time + 1) if road_dict[ti] > 0
    ]
    return {ti: road_dict[ti] for ti in timesteps_to_check}


def alternative_get_new_travel_times(
    roadQueues, roadVDFS, time, arrived_vehicles, time_out_car
):
    new_travel_times = {}
    new_road_queues = {}
    for road in roadQueues.keys():
        arrived_vehicles = arrived_vehicles + [
            car for car in roadQueues[road] if car[3] <= time
        ]
        new_road_queues[road] = [car for car in roadQueues[road] if car[3] > time]
        road_vdf = roadVDFS[road]
        best_known_travel_time = road_vdf(len(new_road_queues[road]))
        max_travel_eta = time + best_known_travel_time
        cars_on_road = len(new_road_queues[road])
        cars_leaving_during_trip = get_cars_leaving_during_trip(
            time_out_car, road, time, max_travel_eta
        )
        cumsum_base = 0
        cars_leaving_cumsum = [
            cumsum_base := cumsum_base + n for n in cars_leaving_during_trip.values()
        ]
        cars_leaving_during_trip_sum = {time: 0} | {
            ti: cars
            for ti, cars in zip(cars_leaving_during_trip.keys(), cars_leaving_cumsum)
        }
        cars_leaving_during_trip_new_tt = {
            ti: ti + road_vdf(cars_on_road - cars_out) - time
            for ti, cars_out in cars_leaving_during_trip_sum.items()
        }
        best_time_out = min(cars_leaving_during_trip_new_tt.values())
        new_travel_times[road] = best_time_out
    return new_travel_times, new_road_queues, arrived_vehicles


def reduced_evaluate_solution(
    solution, car_dist_arrival, timesteps, post_eval=False, seq_decisions=False
):
    """
    solution: list of roads for the vehicles to take, i.e. [1,1,2,2,1,1,2,....,]
    car_dist_arrival: list of arrival times of vehicles, length n, i.e. [1,1,2,3,...,30]
    """
    if len(solution) != len(car_dist_arrival):
        raise Exception("Length of solution and car_dist_arrival must be equal")
    roadQueues = {r: [] for r in set(solution)}
    roadVDFS = {
        1: partial(volume_delay_function, 0.656, 4.8, 15, 20),
        2: partial(volume_delay_function, 0.656, 4.8, 30, 20),
    }
    roadTravelTime = {r: roadVDFS[r](0) for r in roadVDFS.keys()}
    time_out_car = {r: defaultdict(int) for r in roadVDFS.keys()}
    arrived_vehicles = []
    time = 0
    arrival_timestep_dict = Counter(car_dist_arrival)
    sol_deque = deque(solution)
    while not reduced_is_simulation_complete(roadQueues, time, timesteps):
        # get the new vehicles at this timestep

        roadTravelTime, roadQueues, arrived_vehicles = alternative_get_new_travel_times(
            roadQueues, roadVDFS, time, arrived_vehicles, time_out_car
        )

        # Add new vehicles from here
        num_vehicles_arrived = arrival_timestep_dict[time]
        if num_vehicles_arrived is None:
            num_vehicles_arrived = 0
        # just collect the decision of the vehicles
        decisions = [sol_deque.popleft() for _ in range(num_vehicles_arrived)]
        # add vehicles to the new queue
        for decision in decisions:
            roadQueues[decision] = roadQueues[decision] + [
                (
                    decision,
                    time,
                    time + roadTravelTime[decision],
                    time + roadTravelTime[decision],
                )
            ]
            time_out_car[decision][round(time + roadTravelTime[decision])] = (
                time_out_car[decision][round(time + roadTravelTime[decision])] + 1
            )
            if seq_decisions:
                (
                    roadTravelTime,
                    roadQueues,
                    arrived_vehicles,
                ) = alternative_get_new_travel_times(
                    roadQueues, roadVDFS, time, arrived_vehicles, time_out_car
                )

        time += 1
    # Anything which is still in road queue can be added to arrived vehicles
    for road, queue in roadQueues.items():
        arrived_vehicles = arrived_vehicles + [car for car in roadQueues[road]]

    travel_time = [c[3] - c[1] for c in arrived_vehicles]

    if post_eval:
        return (
            nmin(travel_time),
            quantile(travel_time, 0.25),
            mean(travel_time),
            median(travel_time),
            quantile(travel_time, 0.75),
            nmax(travel_time),
            std(travel_time),
            ineq.gini(travel_time),
            ineq.atkinson.index(travel_time, epsilon=0.5),
        )
    return -mean(travel_time)


def convert_to_vot_sol(
    solution,
    car_dist_arrival,
    car_vot_dist,
    timesteps,
    post_eval=False,
    seq_decisions=False,
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
        num_vehicles_arrived = arrival_timestep_dict[time]

        decisions = [sol_deque.popleft() for _ in range(num_vehicles_arrived)]
        cars_arrived = [car_vot_deque.popleft() for _ in range(num_vehicles_arrived)]
        # add vehicles to the new queue
        for decision, vot in zip(decisions, cars_arrived):
            roadQueues[decision] = roadQueues[decision] + [
                (
                    decision,
                    time,
                    vot,
                    time + roadTravelTime[decision],
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
    if post_eval:
        return (
            nmin(time_cost_burden),
            quantile(time_cost_burden, 0.25),
            mean(time_cost_burden),
            median(time_cost_burden),
            quantile(time_cost_burden, 0.75),
            nmax(time_cost_burden),
            std(time_cost_burden),
            ineq.gini(time_cost_burden),
            ineq.atkinson.index(time_cost_burden, epsilon=0.5),
        )
    return -mean(travel_time)

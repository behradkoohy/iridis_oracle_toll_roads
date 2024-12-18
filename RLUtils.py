import numpy as np

from TimeOnlyUtils import get_cars_leaving_during_trip, QueueRanges

def get_sum_of_dict(cars_leaving_dict):
    cumsum_base = 0
    return [cumsum_base := cumsum_base + n for n in cars_leaving_dict.values()]

def get_leaving_cars_sum(cars_leaving_dict, cars_leaving_sum, time):
    cars_leaving_during_trip_sum = {time: 0} | {
        ti: cars
        for ti, cars in zip(cars_leaving_dict.keys(), cars_leaving_sum)
    }
    return cars_leaving_during_trip_sum

def quick_get_new_travel_times(
    roadQueues, roadVDFS, time, arrived_vehicles, time_out_car, queue_manager, vdf_cache
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
        cars_leaving_during_trip, queue_manager = quick_get_cars_leaving_during_trip(
            time_out_car, road, time, max_travel_eta, queue_manager
        )
        cars_leaving_during_trip = {t: c for t, c in cars_leaving_during_trip.items() if c > 0}

        # cumsum_base = 0
        # cars_leaving_cumsum = [
        #     cumsum_base := cumsum_base + n for n in cars_leaving_during_trip.values()
        # ]
        cars_leaving_cumsum = get_sum_of_dict(cars_leaving_during_trip)

        # cars_leaving_during_trip_sum = {time: 0} | {
        #     ti: cars
        #     for ti, cars in zip(cars_leaving_during_trip.keys(), cars_leaving_cumsum)
        # }
        cars_leaving_during_trip_sum = get_leaving_cars_sum(cars_leaving_during_trip, cars_leaving_cumsum, time)

        cars_leaving_during_trip_new_tt = {}
        for ti, cars_out in cars_leaving_during_trip_sum.items():
            # vdf_time = vdf_cache[road].get(cars_on_road - cars_out, None)
            vdf_time = None
            if (cars_on_road - cars_out) in vdf_cache[road]:
            # if vdf_time is None:
                vdf_time = vdf_cache[road][cars_on_road - cars_out]
                vdf_cache[road][cars_on_road - cars_out] = vdf_time
            else:
                vdf_time = road_vdf(cars_on_road - cars_out)
            cars_leaving_during_trip_new_tt[ti] = ti + vdf_time - time
        # cars_leaving_during_trip_new_tt = {
        #     ti: ti + road_vdf(cars_on_road - cars_out) - time
        #     for ti, cars_out in cars_leaving_during_trip_sum.items()
        # }
        best_time_out = min(cars_leaving_during_trip_new_tt.values())
        new_travel_times[road] = best_time_out
    return new_travel_times, new_road_queues, arrived_vehicles, queue_manager, vdf_cache

def quick_get_cars_leaving_during_trip(time_out_car, road, time, max_travel_eta, queue):
    if queue.queues[road] == {}:
        timesteps = get_cars_leaving_during_trip(time_out_car, road, time, max_travel_eta)
        queue.starts[road] = time + 1
        queue.stops[road] = round(max_travel_eta) + 1
        queue.queues[road] = timesteps
        return timesteps, queue
    else:
        new_lowerbound = time + 1
        new_upperbound = round(max_travel_eta) + 1
        old_lowerbound = queue.starts[road]
        old_upperbound = queue.stops[road]
        old_queue = queue.queues[road]
        new_queue = old_queue
        if new_upperbound > old_upperbound:
            new_queue = old_queue | {ti: time_out_car[road][ti] for ti in range(old_upperbound, new_upperbound)}
        if old_upperbound > new_upperbound:
            for x in range(new_upperbound, old_lowerbound):
                new_queue.pop(x, None)
        for x in range(old_lowerbound, new_lowerbound):
            new_queue.pop(x, None)
        queue.queues[road] = new_queue
        """
        CHECK WHAT YOU ARE SAVING THE QUEUE AS WHEN UPDATING THE VALUE
        
        SOMEWHERE YOU ARE OVERWRITING THE CORRECT DATA STRUCTURE AND YOU ARE USING THE WRONG ONE
        
        """
        queue.starts[road] = new_lowerbound
        queue.stops[road] = new_upperbound
        return queue.queues[road], queue
"""
def quicker_get_new_travel_times(
    roadQueues, roadVDFS, time, arrived_vehicles, time_out_car, queue_manager, vdf_cache
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
        cars_leaving_during_trip, queue_manager = quick_get_cars_leaving_during_trip(
            time_out_car, road, time, max_travel_eta, queue_manager
        )
        cars_leaving_during_trip = {t: c for t, c in cars_leaving_during_trip.items() if c > 0}
        cars_leaving_cumsum = np.cumsum(list(cars_leaving_during_trip.values())).tolist()
        cars_leaving_during_trip_sum = dict(
            zip(
                cars_leaving_during_trip.keys(),
                cars_leaving_cumsum
            )
        )
        cars_leaving_during_trip_sum[time] = 0

        left_look = 0
        check_time_keys = list(cars_leaving_during_trip_sum.keys())
        right_look = len(check_time_keys) - 1

        current_best = max_travel_eta - time
        road_vdf_cache = vdf_cache[road]

        while left_look < right_look:
            ti = check_time_keys[left_look]
            cars_out = cars_leaving_during_trip_sum[ti]

            # vdf_time = road_vdf_cache.get(cars_on_road - cars_out, (road_vdf[cars_on_road - cars_out]))
            if not road_vdf_cache.get(cars_on_road - cars_out, False):
                vdf_time = road_vdf(cars_on_road - cars_out)
                vdf_cache[road][cars_on_road - cars_out] = vdf_time
            else:
                vdf_time = road_vdf_cache.get(cars_on_road - cars_out)

            new_tt = ti + vdf_time - time
            current_best = min(new_tt, current_best)

            while (time + current_best < check_time_keys[right_look]):
                right_look -= 1

            left_look += 1

        new_travel_times[road] = current_best

    # if time == 400 or time == 999:
        # breakpoint()

    # print(time, new_travel_times, {x:len(y) for x,y in new_road_queues.items()})
    return new_travel_times, new_road_queues, arrived_vehicles, queue_manager, vdf_cache
"""
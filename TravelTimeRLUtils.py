from queue import PriorityQueue

def pq_get_new_travel_times(time_out_pq, roadVDFs, time, arrived_vehicles):
    """
    We assume that there's a priority queue being held for each road instead of a list of vehicles
    """
    updated_road_travel_times = {}
    for road, queue in time_out_pq.items():
        # remove arrived vehicles from the queue
        while not queue.empty():
            # first, we need to add any vehicles which have arrived to the arrived_vehicles

            next_car = queue.get()[1]
            # print(next_car)
            if next_car[3] <= time:
                arrived_vehicles.append(next_car)
                # print(time, next_car[1], next_car[3] - next_car[1])
                continue
            else:
                queue.put((next_car[3], next_car))
                break

        cars_on_road = queue.qsize()

        best_known_travel_time = roadVDFs[road](cars_on_road)
        cars_still_travelling = []
        while not queue.empty():
            next_car = queue.get()[1]
            # # this is the offset which tells us how many timesteps and how many cars
            # time_offset = next_car[3] - time

            if next_car[3] > best_known_travel_time:
                cars_still_travelling.append(next_car)
                break

            cars_on_road = cars_on_road - 1
            new_travel_time = next_car[3] - time + roadVDFs[road](cars_on_road)
            best_known_travel_time = min(best_known_travel_time, new_travel_time)
            cars_still_travelling.append(next_car)



        updated_road_travel_times[road] = best_known_travel_time

        for car in cars_still_travelling:
            queue.put((car[3], car))

    return updated_road_travel_times, time_out_pq, arrived_vehicles


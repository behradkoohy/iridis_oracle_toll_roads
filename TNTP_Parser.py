import random
import re
from collections import deque
from functools import lru_cache

import numba
import numpy as np
path_to_repo = "/Users/behradkoohy/Development/TransportationNetworks/SiouxFalls/"

def read_trips_file(path):
    metadata = {}
    matrix = {}
    with open(path, 'r') as f:
        # First, parse metadata until <END OF METADATA>
        for line in f:
            line = line.strip()
            if line == "<END OF METADATA>":
                break
            # Example: "<NUMBER OF ZONES> 24"
            if line.startswith("<NUMBER OF ZONES>"):
                metadata["NUMBER OF ZONES"] = int(line.split()[-1])
            elif line.startswith("<TOTAL OD FLOW>"):
                metadata["TOTAL OD FLOW"] = float(line.split()[-1])

        # Now, parse the OD matrix.
        current_origin = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Detect a new Origin block
            if line.startswith("Origin"):
                # e.g., "Origin   1" --> extract the origin number
                parts = line.split()
                current_origin = int(parts[1])
                matrix[current_origin] = {}
            else:
                # Process destination entries
                # Each entry is like: "    2 :    100.0;"
                entries = line.split(";")
                for entry in entries:
                    entry = entry.strip()
                    if entry:  # skip empty entries
                        # Split on the colon to separate destination and flow
                        dest_str, flow_str = entry.split(":")
                        destination = int(dest_str.strip())
                        flow = float(flow_str.strip())
                        matrix[current_origin][destination] = flow

    return metadata, matrix


def sample_trips(od_matrix, simulation_time=86400, dt=60, profile=None, random_state=None):
    """
    Converts an OD flow matrix into a list of trips.

    Parameters:
      od_matrix: dict of dicts, e.g., od_matrix[origin][destination] = total trips (float)
      simulation_time: total simulation time in seconds (default 86400 for 24 hours)
      dt: time step in seconds (default 60 seconds for 1 minute)
      profile: optional function f(t) that gives the fraction of trips at time t (must be normalized)

    Returns:
      A list of dictionaries, each representing a trip with keys "origin", "destination", and "entry_time".
    """
    trips = []
    num_intervals = simulation_time // dt

    for origin in od_matrix:
        for destination, total_trips in od_matrix[origin].items():
            # Skip self-trips if needed, e.g., if origin == destination
            if total_trips <= 0:
                continue

            # Loop over each time interval
            for interval in range(num_intervals):
                t = interval * dt  # entry time (in seconds)

                # Determine expected trips in this interval:
                if profile is None:
                    expected_trips = total_trips / num_intervals
                else:
                    # f(t) should be defined such that sum(f(t)*dt) over the day equals 1
                    expected_trips = total_trips * profile(t) * dt

                # Sample number of trips using a Poisson distribution
                if random_state is not None:
                    n_trips = random_state.poisson(expected_trips)

                else:
                    n_trips = np.random.poisson(expected_trips)

                # For each trip, add an entry to the trip list.
                for _ in range(n_trips):
                    trips.append({
                        "origin": origin,
                        "destination": destination,
                        "entry_time": t  + random_state.randint(0, dt - 1)  # randomize within [t, t+dt)
                        # "entry_time": t  # could randomize within [t, t+dt) if desired
                    })

    # Optionally, sort trips by entry time
    trips.sort(key=lambda x: x["entry_time"])
    return trips


def parse_network_file(file_path):
    """
    Reads the network file and returns metadata (a dictionary) and
    a NumPy array of link data with columns:
    [init_node, term_node, capacity, length, free_flow_time, B, power, speed_limit, toll, link_type]
    """
    metadata = {}
    link_data_lines = []
    meta_done = False

    with open(file_path, 'r') as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue

            # Process metadata until we hit the <END OF METADATA> marker
            if not meta_done:
                if stripped.startswith("<END OF METADATA>"):
                    meta_done = True
                    continue
                if stripped.startswith("<"):
                    # e.g., "<NUMBER OF ZONES> 24"
                    parts = stripped.split(">")
                    if len(parts) >= 2:
                        key = parts[0].strip("<").strip()
                        value = parts[1].strip().split()[0]
                        try:
                            # Convert to int or float as appropriate
                            if '.' in value:
                                metadata[key] = float(value)
                            else:
                                metadata[key] = int(value)
                        except Exception:
                            metadata[key] = value
                continue

            # Once metadata is done, skip comment/header lines (those starting with "~")
            if stripped.startswith("~"):
                continue

            # Process the actual link data lines:
            # Remove any trailing semicolon and extra spaces
            if stripped.endswith(";"):
                stripped = stripped[:-1].strip()
            tokens = stripped.split()
            if tokens:
                link_data_lines.append(tokens)

    # Convert the list of tokens into a NumPy array (all numbers)
    link_data = np.array(link_data_lines, dtype=float)
    # Force node IDs to be integers:
    link_data[:, 0] = link_data[:, 0].astype(int)
    link_data[:, 1] = link_data[:, 1].astype(int)

    return metadata, link_data


def build_connectivity(link_data):
    """
    Given the link data (with columns for origin and destination),
    build:
      - connectivity: dict mapping each origin to a NumPy array of its directly connected destination nodes.
      - link_index: dict mapping (origin, destination) to the row index in link_data.
    """
    connectivity = {}
    link_index = {}

    for i, row in enumerate(link_data):
        origin = int(row[0])
        destination = int(row[1])

        # Build connectivity dictionary:
        if origin not in connectivity:
            connectivity[origin] = []
        connectivity[origin].append(destination)

        # Build mapping for travel time lookup:
        link_index[(origin, destination)] = i

    # Convert lists of neighbors to NumPy arrays for fast, vectorized operations:
    for origin in connectivity:
        connectivity[origin] = np.array(connectivity[origin], dtype=int)

    return connectivity, link_index


def compute_travel_times(flows, free_flow_times, B, capacity, power):
    """
    Vectorized travel time calculation for an array of links.

    Formula:
      travel_time = free_flow_time * (1 + B * (flow/capacity)^power)

    Parameters:
      flows:        numpy array of flows for each link
      free_flow_times, B, capacity, power: numpy arrays (one element per link)

    Returns:
      A numpy array of travel times.
    """
    return free_flow_times * (1 + B * (flows / capacity) ** power)


def get_travel_time(origin, destination, flow, link_data, link_index):
    """
    Given an origin, destination, and flow, return the computed travel time for that link.

    Looks up the corresponding link parameters from link_data via link_index.
    """
    key = (origin, destination)
    if key not in link_index:
        raise ValueError(f"No direct connection from {origin} to {destination}")

    i = link_index[key]
    free_flow_time = link_data[i, 4]
    B = link_data[i, 5]
    capacity = link_data[i, 2]
    power = link_data[i, 6]

    return free_flow_time * (1 + B * (flow / capacity) ** power)


def find_all_paths(origin, destination,
                   connectivity,
                   max_depth=None):
    """
    Recursively enumerates all simple (non-cyclic) paths from origin to destination.

    Parameters:
        origin: The starting node.
        destination: The target node.
        connectivity: A dict mapping each node to a list of its immediate neighbors.
        max_depth: Optional maximum allowed path length (number of nodes).

    Returns:
        A list of paths (each path is a list of node IDs) from origin to destination.
    """

    def _dfs(current, path):
        if current == destination:
            return [path.copy()]
        if max_depth is not None and len(path) > max_depth:
            return []
        paths_found = []
        for neighbor in connectivity.get(current, []):
            if neighbor in path:  # Prevent cycles
                continue
            path.append(neighbor.item())
            paths_found.extend(_dfs(neighbor, path))
            path.pop()  # Backtrack
        return paths_found

    return _dfs(origin, [origin])


def compute_route_metrics(route,
                          link_index,
                          link_data,
                          flow= 50.0):
    """
    Computes the total travel time and cost for a given route.

    Travel time is calculated using:
        travel_time = free_flow_time * (1 + B * (flow / capacity) ** power)

    Cost is assumed to be the sum of tolls (adjust this if your cost function is different).

    Parameters:
        route: List of node IDs representing the route.
        link_index: Dictionary mapping (origin, destination) to an index in link_data.
        link_data: NumPy array of link attributes where columns are:
                   [init_node, term_node, capacity, length, free_flow_time, B, power, speed_limit, toll, link_type]
        flow: Flow value to use in the travel time calculation.

    Returns:
        A tuple (total_travel_time, total_cost).
    """
    total_time = 0.0
    total_cost = 0.0
    for i in range(len(route) - 1):
        orig, dest = route[i], route[i + 1]
        key = (orig, dest)
        if key not in link_index:
            raise ValueError(f"Missing link from {orig} to {dest}.")
        idx = link_index[key]
        free_flow_time = link_data[idx, 4]
        B = link_data[idx, 5]
        capacity = link_data[idx, 2]
        power = link_data[idx, 6]
        toll = link_data[idx, 8]
        travel_time = free_flow_time * (1 + B * (flow / capacity) ** power)
        total_time += travel_time
        total_cost += toll
    return total_time, total_cost


def precompute_routes_for_od_pairs(
        od_pairs,
        connectivity,
        link_index,
        link_data,
        max_depth,
        # flow=50.0
):
    """
    Precomputes all simple routes for each origin-destination pair and evaluates them.

    Parameters:
        od_pairs: List of (origin, destination) tuples.
        connectivity: Dict mapping each node to its list of neighbors.
        link_index: Dict mapping (origin, destination) pairs to row indices in link_data.
        link_data: NumPy array of link attributes.
        max_depth: Optional maximum path length to restrict the search.
        flow: Flow value used for travel time calculations.

    Returns:
        A dictionary mapping each OD pair to a list of tuples:
          { (origin, destination): [ (route, total_travel_time, total_cost), ... ] }
    """
    precomputed_routes = {}
    for origin, destination in od_pairs:
        paths = find_all_paths(origin, destination, connectivity, max_depth)
        # evaluated_routes = []
        # for route in paths:
        #     travel_time, cost = compute_route_metrics(route, link_index, link_data, flow)
        #     evaluated_routes.append((route, travel_time, cost))
        # precomputed_routes[(origin, destination)] = evaluated_routes
        precomputed_routes[(int(origin), int(destination))] = paths
    return precomputed_routes


def bfs_shortest_paths(origin, connectivity):
    """
    Performs a breadth-first search (BFS) from the given origin to compute the
    shortest path (in terms of number of edges) to all reachable nodes.

    Parameters:
        origin: The starting node.
        connectivity: A dictionary mapping each node to a list of its immediate neighbors.

    Returns:
        A dictionary mapping each reachable node to the shortest path (as a list of nodes)
        from the origin.
    """
    visited = {origin: [origin]}
    queue = deque([origin])

    while queue:
        current = queue.popleft()
        for neighbor in connectivity.get(current, []):
            if neighbor not in visited:
                visited[neighbor] = visited[current] + [neighbor]
                queue.append(neighbor)
    return visited

def all_pairs_shortest_paths(connectivity):
    """
    Computes the shortest path (by number of edges) for every pair of nodes in the network.

    Parameters:
        connectivity: A dictionary mapping each node to a list of its immediate neighbors.

    Returns:
        A dictionary where each key is a tuple (origin, destination) and the value is the
        shortest path from origin to destination (as a list of nodes). If a destination is
        unreachable from an origin, it will not appear in the dictionary.
    """
    all_paths = {}
    for origin in connectivity:
        shortest_paths = bfs_shortest_paths(origin, connectivity)
        for destination, path in shortest_paths.items():
            all_paths[(origin, destination)] = path
    return all_paths


def calculate_incoming_flows_per_link(link_data, flows):
    """
    For each link, calculates the total incoming flow to its start node.

    This is done by summing the flows of all links whose destination (term_node)
    matches the start node (init_node) of the link.

    Parameters:
        link_data: A NumPy array of shape (n_links, n_attributes). Column 0 contains the
                   init_node and column 1 contains the term_node.
        flows: A 1D NumPy array of shape (n_links,) where each element is the flow for that link.

    Returns:
        A NumPy array of shape (n_links,) where each element is the total flow arriving at
        the start node of the corresponding link.
    """
    # Extract destination nodes (as integers)
    dest_nodes = link_data[:, 1].astype(int)
    # Determine the maximum node ID (assumes nodes are 1-indexed)
    max_node = dest_nodes.max() + 1
    # Use np.bincount to sum flows for each destination node.
    # Ensure we have an array long enough to include all node IDs.
    incoming_flow_per_node = np.bincount(dest_nodes, weights=flows, minlength=max_node + 1)

    # Extract the start node for each link (as integers)
    start_nodes = link_data[:, 0].astype(int)
    # For each link, look up the sum of flows arriving at its start node.
    result = incoming_flow_per_node[start_nodes]
    result = {start_node: incoming_flow for start_node, incoming_flow in zip(start_nodes, result)}
    return result


def compute_route_travel_time(route, link_index, travel_times_vectorised):
    """
    Computes the total travel time for a given route using a precomputed
    vectorized array of travel times.

    Parameters:
        route: A list of node IDs representing the route (e.g., [3, 4, 11, 10]).
        link_index: A dictionary mapping (origin, destination) tuples to indices in the travel_times_vectorised array.
        travel_times_vectorised: A NumPy array where each element is the travel time for a link,
                                 with indexing corresponding to link_index.

    Returns:
        total_time: The sum of travel times for each link in the route.
    """
    total_time = 0.0
    for i in range(len(route) - 1):
        key = (route[i], route[i + 1])
        if key not in link_index:
            raise ValueError(f"Missing link from {route[i]} to {route[i + 1]}.")
        idx = link_index[key]
        total_time += travel_times_vectorised[idx]
    return total_time

def compute_route_toll_price(route, link_index, route_toll_vectorised):
    """
    Computes the total toll cost for a given route using a precomputed
    vectorized array of travel times.

    Parameters:
        route: A list of node IDs representing the route (e.g., [3, 4, 11, 10]).
        link_index: A dictionary mapping (origin, destination) tuples to indices in the travel_times_vectorised array.
        travel_times_vectorised: A NumPy array where each element is the travel time for a link,
                                 with indexing corresponding to link_index.

    Returns:
        total_time: The sum of travel times for each link in the route.
    """
    total_cost = 0.0
    for i in range(len(route) - 1):
        key = (route[i], route[i + 1])
        if key not in link_index:
            raise ValueError(f"Missing link from {route[i]} to {route[i + 1]}.")
        idx = link_index[key]
        total_cost += route_toll_vectorised[idx]
    return total_cost


def precompute_route_link_indices(route, link_index):
    """
    Given a route (a list of node IDs) and a link_index dictionary,
    precompute and return a NumPy array of link indices corresponding to
    each consecutive node pair in the route.

    Parameters:
      route: List of node IDs representing the route (e.g., [3, 4, 11, 10]).
      link_index: Dictionary mapping (origin, destination) tuples to an index.

    Returns:
      A NumPy array of link indices.
    """
    indices = [link_index[(route[i], route[i + 1])] for i in range(len(route) - 1)]
    return np.array(indices, dtype=int)

def precompute_all_route_indices(routes, link_index):
    """
    Precomputes and caches the link indices for each route.

    Parameters:
      routes: A list of routes, where each route is a list of node IDs.
      link_index: A dictionary mapping (origin, destination) tuples to indices.

    Returns:
      A dictionary mapping each route (represented as a tuple of node IDs) to a NumPy array of link indices.
    """
    route_indices_dict = {}
    for route in routes:
        # Convert route list to tuple to use as a dict key.
        route_key = tuple(route)
        indices = [link_index[(route[i], route[i + 1])] for i in range(len(route) - 1)]
        route_indices_dict[route_key] = np.array(indices, dtype=int)
    return route_indices_dict


@numba.njit
def sum_over_indices(indices, travel_times):
    total = 0.0
    for i in range(len(indices)):
        total += travel_times[indices[i]]
    return total

@numba.njit
def sum_over_both_indices(indices, travel_times, route_costs):
    total_time = 0.0
    total_cost = 0.0
    for i in range(len(indices)):
        total_time += travel_times[indices[i]]
        total_cost += route_costs[indices[i]]
    return total_time, total_cost

def compute_route_travel_time_from_cache(route_key, cached_indices, travel_times_vectorised):
    """
    Computes the total travel time for a route using precomputed link indices.

    Parameters:
      route_key: A tuple representing the route (e.g., (3, 4, 11, 10)).
      cached_indices: A dictionary mapping route tuples to precomputed NumPy arrays of link indices.
      travel_times_vectorised: A NumPy array of travel times for each link.

    Returns:
      Total travel time for the route.
    """
    indices = cached_indices[tuple(route_key)]
    # return travel_times_vectorised[indices].sum()
    return sum_over_indices(indices, travel_times_vectorised)

def compute_route_toll_price_from_cache(route_key, cached_indices, route_toll_vectorised):
    """
    Computes the total toll cost for a route using precomputed link indices.

    Parameters:
      route_key: A tuple representing the route (e.g., (3, 4, 11, 10)).
      cached_indices: A dictionary mapping route tuples to precomputed NumPy arrays of link indices.
      route_toll_vectorised: A NumPy array of toll costs for each link.

    Returns:
      Total toll cost for the route.
    """
    indices = cached_indices[tuple(route_key)]
    # return route_toll_vectorised[indices].sum()
    return sum_over_indices(indices, route_toll_vectorised)

def compute_route_metrics_from_cache(route_key, cached_indices, travel_times_vectorised, route_toll_vectorised):
    """
    Computes the total travel time and toll cost for a route using precomputed link indices.

    Parameters:
      route_key: A tuple representing the route (e.g., (3, 4, 11, 10)).
      cached_indices: A dictionary mapping route tuples to precomputed NumPy arrays of link indices.
      travel_times_vectorised: A NumPy array of travel times for each link.
      route_toll_vectorised: A NumPy array of toll costs for each link.

    Returns:
      A tuple (total_travel_time, total_toll).
    """
    indices = cached_indices[tuple(route_key)]
    total_time, total_toll = sum_over_both_indices(indices, travel_times_vectorised, route_toll_vectorised)
    return total_time, total_toll

if __name__ == "__main__":
    # Define an example route.
    route = [3, 4, 11, 10]

    # Example link_index: mapping (origin, destination) to indices.
    link_index = {
        (3, 4): 0,
        (4, 11): 1,
        (11, 10): 2
    }

    # Example precomputed travel times and toll costs.
    travel_times_vectorised = np.array([4.5, 6.2, 5.1])  # Dummy travel times for each link.
    route_toll_vectorised = np.array([0.5, 1.0, 0.8])  # Dummy toll costs for each link.

    # Compute total travel time and toll cost.
    total_time = compute_route_travel_time(route, link_index, travel_times_vectorised)
    total_toll = compute_route_toll_price(route, link_index, route_toll_vectorised)

    print("Total travel time for the route:", total_time)
    print("Total toll cost for the route:", total_toll)


if __name__ == "__main__":
    # Example link data with columns [init_node, term_node, ...]. Nodes are 1-indexed.
    link_data = np.array([
        [1, 2, 100, 5, 5, 0.15, 4, 0, 0, 1],
        [2, 3, 120, 4, 4, 0.15, 4, 0, 0, 1],
        [4, 1, 110, 6, 6, 0.15, 4, 0, 0, 1],
        [3, 1, 130, 5, 5, 0.15, 4, 0, 0, 1]
    ], dtype=float)

    # Example flows for each link.
    flows = np.array([50, 60, 70, 80], dtype=float)

    # For each link, compute total flow incoming to its start node.
    incoming_flows = calculate_incoming_flows_per_link(link_data, flows)
    print("Incoming flows for each link's start node:", incoming_flows)

# if __name__ == "__main__":
    # metadata, od_matrix = read_trips_file(path_to_repo + "SiouxFalls_trips.tntp")
    # trips = sample_trips(od_matrix, simulation_time=86400)
    # breakpoint()
if __name__ == '__main__':
    # Change this path to where your network file is stored.
    file_path = path_to_repo + "SiouxFalls_net.tntp"

    # 1. Parse the file.
    metadata, link_data = parse_network_file(file_path)
    print("Metadata:")
    print(metadata)

    # 2. Build connectivity and link index mapping.
    connectivity, link_index = build_connectivity(link_data)
    print("\nConnectivity (sample):")
    for node in sorted(connectivity)[:5]:
        print(f"Node {node} connects to {connectivity[node]}")


    # 3. Prepare arrays for vectorized travel time computation.
    # Columns in link_data (assumed order):
    # 0: init_node, 1: term_node, 2: capacity, 3: length, 4: free_flow_time,
    # 5: B, 6: power, 7: speed_limit, 8: toll, 9: link_type
    free_flow_times = link_data[:, 4]
    B_array = link_data[:, 5]
    capacity_array = link_data[:, 2]
    power_array = link_data[:, 6]

    # 4. Assume a vector of flows for each link (for example, 0 vehicles on each link)
    flows = np.full(free_flow_times.shape, 0.0)

    # 5. Compute travel times for all links at once using vectorization.
    travel_times_vectorized = compute_travel_times(flows, free_flow_times, B_array, capacity_array, power_array)
    print("\nVectorized travel times for each link:")
    print(travel_times_vectorized)

    # 6. For a single link lookup, e.g., from node 1 to node 2 with a flow of 50:
    tt = get_travel_time(1, 2, 50.0, link_data, link_index)
    print(f"\nTravel time from node 1 to 2 with flow 50: {tt}")

    od_pairs = [(1, 20)]
    max_depth = 9
    flow = 0

    all_paths = all_pairs_shortest_paths(connectivity)

    for (origin, destination), path in all_paths.items():
        print(f"Shortest path from {origin} to {destination}: {len(path)}")

    # precomputed = precompute_routes_for_od_pairs(od_pairs, connectivity, link_index, link_data, max_depth, flow)
    # for (origin, destination), routes in precomputed.items():
    #     print(f"Routes from {origin} to {destination}:")
    #     for route, travel_time, cost in routes:
    #         print(f"  Route: {route}, Travel Time: {travel_time:.2f}, Cost: {cost:.2f}")

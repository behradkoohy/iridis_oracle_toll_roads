import argparse
import numpy.random as nprand
import numpy as np
from numpy import mean, quantile, median, std
from numpy import min as nmin
from numpy import max as nmax
import inequalipy as ineq
from collections import Counter, defaultdict, deque
from functools import partial
import pyswarms as ps
from scipy.stats import gmean

from TrackedGlobalBestPSO import GlobalBestPSO

from torch.utils.tensorboard import SummaryWriter

from multiprocessing import Pool
from functools import partial

from DiscretePySwarms import IntOptimizerPSO
from RLUtils import quick_get_new_travel_times
from TimeOnlyUtils import volume_delay_function, QueueRanges
from TimestepPriceUtils import generate_utility_funct, quantal_decision


def generate_car_time_distribution(n_cars, timesteps, beta_dist_alpha=5, beta_dist_beta=5, timeseed=None):
    nprand.seed(timeseed)
    car_dist_norm = nprand.beta(
        beta_dist_alpha,
        beta_dist_beta,
        size=n_cars,
    )
    car_dist_arrival = list(
        map(
            lambda z: round(
                (z - min(car_dist_norm))
                / (max(car_dist_norm) - min(car_dist_norm))
                * timesteps
            ),
            car_dist_norm,
        )
    )
    return car_dist_arrival

def generate_car_vot_distribution(n_cars, car_vot_lowerbound=0.0, car_vot_upperbound=1.0, votseed=None):
    nprand.seed(votseed)
    car_vot = nprand.uniform(
        car_vot_lowerbound, car_vot_upperbound, n_cars
    )
    return car_vot

def timestep_price_update(road_prices, timestep_action):
    pricing_dict = {
        -1: lambda x: max(x - 1, 1),
        0: lambda x: x,
        1: lambda x: min(x + 1, 125),
    }
    new_road_prices = {r: pricing_dict[timestep_action[r]](road_prices[r]) for r in road_prices.keys()}
    return new_road_prices

def linear_price_update(road_prices, timestep_action):
    return road_prices

update_price_funct = {
    "Linear": linear_price_update,
}

def create_lin_funct(m,c):
    return lambda x: (m * x) + c

def evaluate_solution(
    solution,
    car_dist_arrival,
    car_vot_dist,
    timesteps,
    args,
    road_price_upt=linear_price_update,
    post_eval=False,
    seq_decisions=False,
    optimise_for="TravelTime",
):
    """
    solution: list of roads for the vehicles to take, i.e. [1,1,2,2,1,1,2,....,]
    car_dist_arrival: list of arrival times of vehicles, length n, i.e. [1,1,2,3,...,30]
    """

    roadQueues = {r: [] for r in [0, 1]}
    roadVDFS = {
        0: partial(volume_delay_function, 0.656, 4.8, args.road_0_cap, args.road_0_t0),
        1: partial(volume_delay_function, 0.656, 4.8, args.road_1_cap, args.road_1_t0),
    }
    roadTravelTime = {r: roadVDFS[r](0) for r in roadVDFS.keys()}
    arrival_timestep_dict = Counter(car_dist_arrival)
    arrived_vehicles = []

    sol_deque = deque(solution)
    car_vot_deque = deque(car_vot_dist)

    if args.actions == "Linear":
        # roadPrices_funct = {r: lambda x: ((solution[(2*n)] * x) + solution[(2*n)+1]) for n, r in enumerate(roadVDFS.keys())}
        roadPrices_funct = {r: create_lin_funct(solution[(2*n)], solution[(2*n)+1]) for n, r in enumerate(roadVDFS.keys())}
        roadPrices = {r: roadPrices_funct[r](0) for r in roadVDFS.keys()}
        # breakpoint()
    elif args.actions == "Fixed":
        # roadPrices_funct = {r: [sol_deque.popleft() for _ in roadVDFS.keys()] for r in roadVDFS.keys()}
        # roadPrices = {r: roadPrices_funct[r](0) for r in roadVDFS.keys()}
        roadPrices = {r: sol_deque.popleft() for r in roadVDFS.keys()}
    elif args.actions == "Unbound":
        # roadPrices = {r: [sol_deque.popleft() for _ in roadVDFS.keys()] for r in roadVDFS.keys()}
        roadPrices = {r: 50 for r in roadVDFS.keys()}
    elif args.actions == "Timestep":
        roadPrices = {r: 50 for r in roadVDFS.keys()}
    elif args.actions == "Free":
        roadPrices = {r: 0 for r in roadVDFS.keys()}
    else:
        print('using alternative pricing')
        roadPrices = {r: 0 for r in roadVDFS.keys()}

    time_out_car = {r: defaultdict(int) for r in roadVDFS.keys()}

    arrived_vehicles = []
    time = 0

    queue_ranges = QueueRanges()
    vdf_cache = {agent: {} for agent in range(2)}

    while not time >= timesteps + 1:
        # get the new vehicles at this timestep
        roadTravelTime, roadQueues, arrived_vehicles, queue_ranges, vdf_cache = quick_get_new_travel_times(
            roadQueues, roadVDFS, time, arrived_vehicles, time_out_car, queue_ranges, vdf_cache
        )

        # we pass in the actions for this timestep and the road prices, and we get back the new road prices.
        if args.actions == 'Free':
            roadPrices = {r: 0 for r in roadVDFS.keys()}
        elif args.actions != "Linear":
            roadPrices = road_price_upt(roadPrices, [sol_deque.popleft() for _ in roadVDFS.keys()] if args.actions in ["Timestep", "Unbound"] else [])
        elif args.actions == "Linear":
            roadPrices = {r: roadPrices_funct[r](len(roadQueues[r])) for r in roadPrices_funct.keys()}
        roadPrices = {r: min(max(x, 1), 125) for r, x in roadPrices.items()}
        # Add new vehicles from here
        num_vehicles_arrived = arrival_timestep_dict[time]
        # print(roadTravelTime, {r:len(x) for r,x in roadQueues.items()}, roadPrices)
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
                roadTravelTime, roadQueues, arrived_vehicles, queue_ranges, vdf_cache = quick_get_new_travel_times(
                    roadQueues, roadVDFS, time, arrived_vehicles, time_out_car, queue_ranges, vdf_cache
                )
        time += 1
    for road, queue in roadQueues.items():
        arrived_vehicles = arrived_vehicles + [car for car in roadQueues[road]]
    travel_time = [c[3] - c[1] for c in arrived_vehicles]
    time_cost_burden = [(c[3] - c[1]) * c[2] for c in arrived_vehicles]
    combined_cost = [((c[3] - c[1]) * c[2]) + c[4] for c in arrived_vehicles]
    profit = [c[4] for c in arrived_vehicles]
    if post_eval:
        return (
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
            (
                nmin(profit),
                quantile(profit, 0.25),
                mean(profit),
                median(profit),
                quantile(profit, 0.75),
                nmax(profit),
                std(profit),
                ineq.gini(profit),
                ineq.atkinson.index(profit, epsilon=0.5),
            ),
        )
    return {
        "TravelTime": mean(travel_time),
        "SocialCost": mean(time_cost_burden),
        "CombinedCost": mean(combined_cost),
        "Profit": -mean(profit),
    }[optimise_for]


# def evaluate_all_scenarios(car_dists, vot_dists, sol, post_eval=False):
#     # print("EAS CALL")
#     # pool = Pool()
#     solution_scores = []
#
#     for car_dist, vot_dist in zip(car_dists, vot_dists):
#         solution_scores.append(evaluate_solution(
#             solution=sol,
#             car_dist_arrival=car_dist,
#             car_vot_dist=vot_dist,
#             timesteps=args.timesteps,
#             args=args,
#             seq_decisions=False,
#             optimise_for=args.optimise,
#             post_eval=post_eval,
#         ))
#     return gmean(solution_scores)
def apply(f, x):
    return f(x)

def evaluate_all_scenarios(partial_evals, sol, post_eval=False):
    # print("EAS CALL")
    pool = Pool()
    # solution_scores = []
    apply_partial = partial(apply, x=sol)
    solution_scores = pool.map(apply_partial, partial_evals)

    # solution_scores = list(map(lambda x: x(sol), partial_evals))
    # solution_scores = list(solution_scores)
    # for car_dist, vot_dist in zip(car_dists, vot_dists):
    #     solution_scores.append(evaluate_solution(
    #         solution=sol,
    #         car_dist_arrival=car_dist,
    #         car_vot_dist=vot_dist,
    #         timesteps=args.timesteps,
    #         args=args,
    #         seq_decisions=False,
    #         optimise_for=args.optimise,
    #         post_eval=post_eval,
    #     ))
    # breakpoint()
    return gmean(list(solution_scores))



def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, help="The number of timesteps for the simulation", default=1000)
    # parser.add_argument("--n_cars", type=int, help="The number of cars in the simulation", default=632000)
    parser.add_argument("--optimise", type=str, help="Select the evaluation function for the model", choices=['TravelTime', 'SocialCost', 'CombinedCost', 'Profit'], default='TravelTime')
    parser.add_argument("--actions", type=str, help="Select the action space of the agent", choices=['Linear'], default='Linear')
    parser.add_argument("--VOTSeed", type=int, help="The VOT seed value", default=1)
    parser.add_argument("--TIMESeed", type=int, help="The TIME seed value", default=1)
    parser.add_argument("--beta_alpha", type=int, help="The alpha value for the beta distribution", default=5)
    parser.add_argument("--beta_beta", type=int, help="The beta value for the beta distribution", default=5)
    parser.add_argument("--car_vot_lowerbound", type=float, help="The lower bound for the car VOT distribution", default=0.0)
    parser.add_argument("--car_vot_upperbound", type=float, help="The upper bound for the car VOT distribution", default=1.0)
    parser.add_argument("--n_iterations", type=int, help="The number of iterations for the PSO algorithm", default=250)
    parser.add_argument("--n_particles", type=int, help="The number of particles in the PSO algorithm", default=5)
    parser.add_argument("--track", type=bool, help="Track the experiment with Weights and Biases", default=True)
    parser.add_argument("--road_0_t0", type=int, help="Free flow travel time of road 0, default 15", default=15)
    parser.add_argument("--road_1_t0", type=int, help="Free flow travel time of road 1, default 30", default=30)
    parser.add_argument("--road_0_cap", type=int, help="Free flow capacity of road 0, default 20", default=20)
    parser.add_argument("--road_1_cap", type=int, help="Free flow capacity of road 1, default 20", default=20)
    return parser.parse_args()
    # fmt: on

def run_exp(args):
    # run_name = '_'.join([str(args.timesteps), str(args.n_cars), args.optimise, args.actions, str(args.VOTSeed), str(args.TIMESeed)])
    run_name = 'LINEAR_RUN_ALL_SCEN_OPT'
    if args.track:
        import wandb

        run = wandb.init(
            project="MMRP_PSO",
            entity=None,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    car_dist_arrival_range = []
    car_vot_arrival_range = []
    partial_eval_functs = []
    for n_cars in range(500, 1001, 50):
        for time_seed in range(10):
            car_dist_arrival = generate_car_time_distribution(
                n_cars,
                args.timesteps,
                timeseed=time_seed,
                beta_dist_alpha=args.beta_alpha,
                beta_dist_beta=args.beta_beta
            )
            car_dist_arrival_range.append(car_dist_arrival)
            car_vot_arrival = generate_car_vot_distribution(
                n_cars,
                votseed=time_seed,
                car_vot_lowerbound=args.car_vot_lowerbound,
                car_vot_upperbound=args.car_vot_upperbound
            )
            car_vot_arrival_range.append(car_vot_arrival)
            partial_eval_functs.append(
                partial(evaluate_solution,
                        car_dist_arrival=car_dist_arrival,
                        car_vot_dist=car_vot_arrival,
                        timesteps=args.timesteps,
                        args=args,
                        seq_decisions=False,
                        optimise_for=args.optimise,
                        post_eval=False,)
            )

    def objective_function(solution):
        # solution = [list(map(discrete_activate_funct, sol)) for sol in solution]
        # if args.actions == "Timestep":
        #     solution = [list(map(discrete_activate_funct, sol)) for sol in solution]
        score = [
            # evaluate_all_scenarios(
            #     car_dist_arrival_range,
            #     car_vot_arrival_range,
            #     sol,
            # )
            evaluate_all_scenarios(partial_eval_functs, sol)
            for sol in solution
        ]
        return score

    if args.actions == 'Linear':
        # min_bound = [-100 for _ in range(4)]
        # max_bound = [100 for _ in range(4)]
        min_bound = [0, 1, 0, 1]
        max_bound = [50, 125, 50, 125]

    bounds = (min_bound, max_bound)
    # options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
    options = {"c1": 2.05, "c2": 2.05, "w": 0.729}
    # NOTE: Unneeded but remmenant of old code
    if args.actions == 'Linear':
        optimizer = GlobalBestPSO(
            n_particles=args.n_particles, dimensions=4, options=options, bounds=bounds,
            track=writer if args.track else None,
            wandb=wandb if args.track else None,
        )

    cost, pos = optimizer.optimize(objective_function, iters=args.n_iterations)
    # tt_eval, sc_eval, cc_eval, pr_eval = evaluate_solution(
    #     pos,
    #     car_dist_arrival,
    #     car_vot_arrival,
    #     road_price_upt,
    #     args.timesteps,
    #     args,
    #     seq_decisions=False,
    #     post_eval=True,
    #     optimise_for=args.optimise,
    # )
    print(cost, pos)
    # if args.track:
    #     for name, mode in zip(['travel_time', 'social_cost', 'combined_cost', 'profit'], [tt_eval, sc_eval, cc_eval, pr_eval]):
    #         writer.add_scalar(f"{name}/{name}_min", mode[0])
    #         writer.add_scalar(f"{name}/{name}_q1", mode[1])
    #         writer.add_scalar(f"{name}/{name}_mean", mode[2])
    #         writer.add_scalar(f"{name}/{name}_median", mode[3])
    #         writer.add_scalar(f"{name}/{name}_q3", mode[4])
    #         writer.add_scalar(f"{name}/{name}_max", mode[5])
    #         writer.add_scalar(f"{name}/{name}_std", mode[6])
    #         writer.add_scalar(f"{name}/{name}_gini", mode[7])
    #         writer.add_scalar(f"{name}/{name}_atkinson", mode[8])

    if args.track:
        print("eval begin")
        for eval_n_cars in range(500, 1001, 25):
            print(eval_n_cars)
            trained_agent_means_tt = []
            trained_agent_means_sc = []
            trained_agent_means_cc = []
            trained_agent_means_pr = []
            # for time_seed in range(10):
            for time_seed in range(10):
                car_dist_arrival = generate_car_time_distribution(
                    eval_n_cars,
                    args.timesteps,
                    timeseed=time_seed,
                    beta_dist_alpha=args.beta_alpha,
                    beta_dist_beta=args.beta_beta
                )
                car_vot_arrival = generate_car_vot_distribution(
                    eval_n_cars,
                    votseed=time_seed,
                    car_vot_lowerbound=args.car_vot_lowerbound,
                    car_vot_upperbound=args.car_vot_upperbound
                )
                tt_eval, sc_eval, cc_eval, pr_eval = evaluate_solution(
                    pos,
                    car_dist_arrival = car_dist_arrival,
                    car_vot_dist = car_vot_arrival,
                    timesteps = args.timesteps,
                    args = args,
                    seq_decisions = False,
                    optimise_for = args.optimise,
                    post_eval = True
                )
                trained_agent_means_tt.append(tt_eval)
                trained_agent_means_sc.append(sc_eval)
                trained_agent_means_cc.append(cc_eval)
                trained_agent_means_pr.append(pr_eval)
            for name, mode in zip(
                    ['travel_time', 'social_cost', 'combined_cost', 'profit'],
                    [trained_agent_means_tt, trained_agent_means_sc, trained_agent_means_cc, trained_agent_means_pr]
            ):
                writer.add_scalar(f"{name}/{name}_min", np.mean([x[0] for x in mode]))
                writer.add_scalar(f"{eval_n_cars}/{name}_q1", np.mean([x[1] for x in mode]))
                writer.add_scalar(f"{eval_n_cars}/{name}_mean", np.mean([x[2] for x in mode]))
                writer.add_scalar(f"{eval_n_cars}/{name}_median", np.mean([x[3] for x in mode]))
                writer.add_scalar(f"{eval_n_cars}/{name}_q3", np.mean([x[4] for x in mode]))
                writer.add_scalar(f"{eval_n_cars}/{name}_max", np.mean([x[5] for x in mode]))
                writer.add_scalar(f"{eval_n_cars}/{name}_std", np.mean([x[6] for x in mode]))
                writer.add_scalar(f"{eval_n_cars}/{name}_gini", np.mean([x[7] for x in mode]))
                writer.add_scalar(f"{eval_n_cars}/{name}_atkinson", np.mean([x[8] for x in mode]))





if __name__ == "__main__":
    args = parse_args()
    args.VOTSeed = args.TIMESeed
    run_exp(args)

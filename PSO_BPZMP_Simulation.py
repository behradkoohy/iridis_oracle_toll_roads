from multiprocessing import Pool

import numpy as np
from matplotlib import pyplot as plt
from pyswarms.utils.plotters import plot_cost_history
from scipy.stats import gmean
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from PSOBigPZEnv import TNTPParallelEnv
import argparse

from TrackedGlobalBestPSO import GlobalBestPSO
# from pyswarms.single import GlobalBestPSO


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="PSO for BigPZMP Simulation")
    # parser.add_argument('--num_agents', type=int, default=5, help='Number of agents in the environment')
    # parser.add_argument("--tntp_path", type=str,
                        # default="/Users/behradkoohy/Development/TransportationNetworks/Small-Seq-Example/Small-Seq-Example",
                        # help="Path to TNTP files")
    # parser.add_argument("--tntp_path", type=str,
    #                     default="/Users/behradkoohy/Development/TransportationNetworks/SiouxFalls/SiouxFalls",
    #                     help="Path to TNTP files")
    parser.add_argument("--tntp_path", type=str,
                        default="C:\\Users\\Behrad\\PycharmProjects\\iridis_oracle_toll_roads\\TransportationNetworks-master\\Small-Seq-Example\\Small-Seq-Example",
                        help="Path to TNTP files")
    parser.add_argument("--timesteps", type=int, default=10,
                        help="total episodes of the experiments")
    parser.add_argument('--num_iterations', type=int, default=180, help='Number of iterations for PSO')
    parser.add_argument('--num_particles', type=int, default=10, help='Number of particles in PSO')
    parser.add_argument("--track", type=bool, help="Track the experiment with Weights and Biases", default=True)
    parser.add_argument(
        "--pricing_mode",
        type=str,
        choices=["step", "linear", "fixed", "unbound"],
        default="fixed",
        help="Pricing mode to optimize: 'step', 'linear', 'fixed' or 'unbound'."
    )
    parser.add_argument(
        "--vot_dist",
        type=str,
        choices=["normal", "dagum", "uniform"],
        default="uniform",
        help="Choice of vot distribution to sample"
    )
    parser.add_argument("--n_seeds_train", type=int, default=10, help="Number of random seeds for evaluation")
    parser.add_argument("--n_seeds_eval", type=int, default=50, help="Number of random seeds for evaluation")
    args = parser.parse_args()
    return args


def evaluate_solution(solution, env, seed, args):
    """
    Evaluate the solution in the environment.
    """
    total_reward = 0
    next_obs, info = env.reset(seed=seed)

    # if args.timesteps * env.num_agents != len(solution):
    #     raise ValueError(f"Solution length {len(solution)} does not match expected length {args.timesteps * env.num_agents}.")
    while env.agents:
        # if we run out of actions, we maintain the same price for the rest of the time
        if len(solution) < len(env.agents):
            actions = {agent_id: 0 for agent_id in env.agents}
        else:
            actions = {agent_id: solution.pop(0) for agent_id in env.agents}
        # print(env.time, actions)
        next_obs, rewards, terms, truncs, infos = env.step(actions)
        total_reward += sum(rewards.values())
        # Check if the environment has reached the end of the episode
        if env.is_simulation_complete():
            return (sum(env.travel_time))/ len(env.travel_time) if env.travel_time else 0


def evaluate_across_seeds(solutions):
    """
    Vectorized evaluation: `solutions` is an array of shape (n_particles, dimension).
    Returns an array of shape (n_particles,) containing the geometric-mean score
    of each candidate solution over 50 random seeds.
    """
    args = parse_args()
    seeds = range(args.n_seeds_train)
    n_particles = solutions.shape[0]
    costs = np.zeros(n_particles)

    for i, particle in enumerate(solutions):
        # evaluate this one solution across all seeds
        seed_results = []
        for seed in seeds:
            env = TNTPParallelEnv(
                tntp_path=args.tntp_path,
                timesteps=args.timesteps,
                pricing_mode=args.pricing_mode,
                pricing_params=particle.tolist(),
                seed=seed,
                vot_dist=args.vot_dist,
            )
            sol_copy = particle.tolist()
            score = evaluate_solution(sol_copy, env, seed, args)
            seed_results.append(score)
        # geometric mean across seeds
        costs[i] = gmean(seed_results)

    return costs

def post_evaluate_across_seeds(solutions):
    """
    Vectorized evaluation: `solutions` is an array of shape (n_particles, dimension).
    Returns an array of shape (n_particles,) containing the geometric-mean score
    of each candidate solution over 50 random seeds.
    """
    args = parse_args()
    seeds = range(20)
    n_particles = solutions.shape[0]
    costs = np.zeros(n_particles)

    for i, particle in enumerate(solutions):
        # evaluate this one solution across all seeds
        seed_results = []
        for seed in seeds:
            env = TNTPParallelEnv(
                tntp_path=args.tntp_path,
                timesteps=args.timesteps,
                pricing_mode=args.pricing_mode,
                pricing_params=particle.tolist(),
                seed=seed,
                vot_dist=args.vot_dist,
            )
            sol_copy = particle.tolist()
            score = evaluate_solution(sol_copy, env, seed, args)
            seed_results.append(score)
        # geometric mean across seeds
        costs[i] = gmean(seed_results)

    return costs

def run_exp(args):
    """
    Run the experiment with the given arguments.
    """
    # Initialize the environment with pricing_mode and empty pricing_params to get env properties
    env = TNTPParallelEnv(
        tntp_path=args.tntp_path,
        timesteps=args.timesteps,
        pricing_mode=args.pricing_mode,
        pricing_params=[],
        seed=1,
        vot_dist=args.vot_dist,
    )
    # Compute the PSO dimension based on pricing_mode
    if args.pricing_mode == "step":
        dimensions = args.timesteps * env.num_links
        bounds = (
            list(-1 for _ in range(dimensions)),
            list(1 for _ in range(dimensions))
        )
    elif args.pricing_mode == "linear":
        dimensions = 2 * env.num_links
        bounds = (
            list(-100 for _ in range(dimensions)),
            list(100 for _ in range(dimensions))
        )
    elif args.pricing_mode == "fixed":
        dimensions = env.num_links
        bounds = (
            list(1 for _ in range(dimensions)),
            list(130 for _ in range(dimensions))
        )
    elif args.pricing_mode == "unbound":
        dimensions = args.timesteps * env.num_links
        bounds = (
            list(1 for _ in range(dimensions)),
            list(125 for _ in range(dimensions))
        )
    else:
        raise ValueError(f"Unknown pricing_mode {args.pricing_mode}")

    # # from running fixed pricing, we have a best solution for unbound pricing
    # if args.pricing_mode == "unbound":
    #     # this is a solution that was found by running fixed pricing
    #     # it is not the best solution, but it is a good starting point
    #     # for unbound pricing
    #     fixed_pos_sol = [52.48394325453418, 108.48906216957744, 30.472827337496014, 82.4303810897569, 80.76420061280888, 124.62517753339074, 32.85574568000811, 24.130522834072735, 105.93401716192132, 114.87998126653594, 60.39447729287144, 53.918611093237814, 23.656357642350546, 120.43856193398432, 51.806142616218494, 82.17998843813349, 35.429503133859725, 25.20867456371016]
    #     # first, we need to expand this to the correct dimensions
    #     init_sol = []
    #     for sol in fixed_pos_sol:
    #         # we want to repeat each value for the number of timesteps
    #         init_sol.extend([sol] * args.timesteps)
    #     # now we have a solution of length timesteps * num_links
    #     # we need to add some noise to it
    #     noise = np.random.uniform(-5, 5, size=len(init_sol))  # Generate noise
    #     init_sols = []
    #     for i in range(args.num_particles):
    #         # Generate a new solution with noise for each particle
    #         init_sol_with_noise = np.array(init_sol) + noise
    #         # Ensure the solution is within bounds
    #         init_sol_with_noise = np.clip(init_sol_with_noise, bounds[0], bounds[1])
    #         init_sols.append(init_sol_with_noise)
    #
    #     # Convert to a numpy array
    #     init_sol = np.array(init_sols)
    #     # we will use this as the initial position for the PSO



    options = {"c1": 2.05, "c2": 2.05, "w": 0.729}
    optimizer = GlobalBestPSO(
        n_particles=args.num_particles,
        dimensions=dimensions,
        options=options,
        bounds=bounds,
        track=writer if args.track else None,
        wandb=wandb if args.track else None,
        # init_pos=init_sol if args.pricing_mode == "unbound" else None
    )
    # cost, pos = optimizer.optimize(evaluate_solution, iters=args.num_iterations)
    # we want to evaluate the solution across multiple seeds
    cost, pos = optimizer.optimize(evaluate_across_seeds, iters=args.num_iterations, n_processes=5)
    # cost, pos = optimizer.optimize(evaluate_across_seeds, iters=args.num_iterations)
    # Convert solution vector to list
    pos_not_list = pos
    pos = pos.tolist()
    # Only round to integers for the discrete “step” mode
    if args.pricing_mode == "step":
        pos = [int(round(x)) for x in pos]
    print(f"Optimized cost: {cost}, Optimized position: {pos}")
    # Final evaluation: Evaluate the optimized solution in the environment
    # env = TNTPParallelEnv(
    #     tntp_path=args.tntp_path,
    #     timesteps=args.timesteps,
    #     pricing_mode=args.pricing_mode,
    #     pricing_params=pos,
    #     seed=1
    # )
    # total_reward = evaluate_solution(pos.copy(), env, seed=1, args=args)
    # print(f"Total reward for the optimized solution: {total_reward}")
    if args.track:
        wandb.run.summary["opt_cost_hist"] = optimizer.cost_history
        wandb.run.summary["training_cost"] = cost



    plot_cost_history(cost_history=optimizer.cost_history)
    plt.show()
    if args.track:
        print("evaluating begin")
        trained_agent_means_tt = []
        trained_agent_gini_tt = []
        trained_agent_means_sc = []
        trained_agent_gini_sc = []
        trained_agent_means_cc = []
        trained_agent_gini_cc = []
        for seed in trange(args.n_seeds_eval):
            env = TNTPParallelEnv(
                tntp_path=args.tntp_path,
                timesteps=args.timesteps,
                pricing_mode=args.pricing_mode,
                pricing_params=pos_not_list.tolist(),
                seed=seed,
                vot_dist=args.vot_dist,
            )
            # sol_copy = pos.tolist()
            score = evaluate_solution(pos, env, seed, args)
            trained_agent_means_tt.append(gmean(env.travel_time))
            trained_agent_gini_tt.append(env.gini_tt)
            trained_agent_means_sc.append(gmean(env.time_cost_burden))
            trained_agent_gini_sc.append(env.gini_sc)
            trained_agent_means_cc.append(gmean(env.combined_cost))
            trained_agent_gini_cc.append(env.gini_cc)

        wandb.run.summary["trained_travel_times"] = trained_agent_means_tt
        wandb.run.summary["gini_travel_time"] = trained_agent_gini_tt
        wandb.run.summary["trained_social_costs"] = trained_agent_means_sc
        wandb.run.summary["gini_social_costs"] = trained_agent_gini_sc
        wandb.run.summary["trained_combined_costs"] = trained_agent_means_cc
        wandb.run.summary["gini_combined_costs"] = trained_agent_gini_cc
        print(trained_agent_means_tt, trained_agent_means_sc, trained_agent_means_cc)


if __name__ == "__main__":

    args = parse_args()

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

    # Initialize the environment
    # env = TNTPParallelEnv(tntp_path=args.tntp_path, timesteps=args.timesteps, seed=1)

    # Run the experiment
    run_exp(args)

    # # Example solution: a list of actions for each agent
    # # This should be replaced with the actual PSO solution
    # example_solution = [0 for _ in range((args.timesteps) * env.num_agents)]
    #
    # # Evaluate the example solution
    # total_reward = evaluate_solution(example_solution, env, seed=1)
    #
    # print(f"Total reward for the example solution: {total_reward}")
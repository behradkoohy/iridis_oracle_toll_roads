from Experiment import Experiment

import pyswarms as ps
from TimestepPriceUtils import evaluate_solution_linearprice
from PSO_CONFIG import *

# TODO: you need to change some parameters to make the linear cost model work.


class ParticleSwarmLinearPriceCC(Experiment):
    def __init__(self, db_path, ID, model_name, description, votseed, timeseed):
        super().__init__(db_path, ID, model_name, description, votseed, timeseed)
        self.bound = 1
        self.run_exp()

    def run_exp(self):
        car_dist_arrival = self.generate_car_time_distribution()
        car_vot_arrival = self.generate_car_vot_distribution()

        def objective_function(solution):
            solution = [{1: (sol[0], sol[1]), 2: (sol[2], sol[3])} for sol in solution]
            score = [
                -evaluate_solution_linearprice(
                    sol,
                    car_dist_arrival,
                    car_vot_arrival,
                    self.timesteps,
                    seq_decisions=True,
                    optimise_for="CombinedCost",
                )
                for sol in solution
            ]
            return score

        max_bound = [100 for _ in range(4)]
        min_bound = [0 for _ in range(4)]
        bounds = (min_bound, max_bound)
        options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
        optimizer = ps.single.GlobalBestPSO(
            n_particles=N_PARTICLES, dimensions=4, options=options, bounds=bounds
        )
        cost, pos = optimizer.optimize(objective_function, iters=N_ITERATIONS)
        travel_time, social_cost, combined_cost = evaluate_solution_linearprice(
            {1: (pos[0], pos[1]), 2: (pos[2], pos[3])},
            car_dist_arrival,
            car_vot_arrival,
            self.timesteps,
            seq_decisions=True,
            optimise_for="CombinedCost",
            post_eval=True,
        )
        self.write_results("TravelTime", *travel_time)
        self.write_results("SocialCost", *social_cost)
        self.write_results("CombinedCost", *combined_cost)
        print(cost, pos)

    # def main(self):


if __name__ == "__main__":
    pso = ParticleSwarmLinearPriceCC("", 0, 0, 0, 0, 0)

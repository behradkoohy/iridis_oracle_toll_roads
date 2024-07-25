from Experiment import Experiment

import pyswarms as ps
from TimestepPriceUtils import evaluate_solution_fixedprice
from PSO_CONFIG import *


class ParticleSwarmFixedPriceTT(Experiment):
    def __init__(self, db_path, ID, model_name, description, votseed, timeseed):
        super().__init__(db_path, ID, model_name, description, votseed, timeseed)
        self.bound = 1
        self.run_exp()

    def run_exp(self):
        car_dist_arrival = self.generate_car_time_distribution()
        car_vot_arrival = self.generate_car_vot_distribution()

        def objective_function(solution):
            solution = [{1: sol[0], 2: sol[1]} for sol in solution]
            score = [
                -evaluate_solution_fixedprice(
                    sol,
                    car_dist_arrival,
                    car_vot_arrival,
                    self.timesteps,
                    seq_decisions=True,
                    optimise_for="TravelTime",
                )
                for sol in solution
            ]
            return score

        max_bound = [100 for _ in range(2)]
        min_bound = [0 for _ in range(2)]
        bounds = (min_bound, max_bound)
        options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
        optimizer = ps.single.GlobalBestPSO(
            n_particles=N_PARTICLES, dimensions=2, options=options, bounds=bounds
        )
        cost, pos = optimizer.optimize(objective_function, iters=N_ITERATIONS)
        travel_time, social_cost, combined_cost = evaluate_solution_fixedprice(
            {1: pos[0], 2: pos[1]},
            car_dist_arrival,
            car_vot_arrival,
            self.timesteps,
            seq_decisions=True,
            optimise_for="TravelTime",
            post_eval=True,
        )
        self.write_results("TravelTime", *travel_time)
        self.write_results("SocialCost", *social_cost)
        self.write_results("CombinedCost", *combined_cost)
        print(cost, pos)

    # def main(self):


if __name__ == "__main__":
    pso = ParticleSwarmFixedPriceTT("", 0, 0, 0, 0, 0)

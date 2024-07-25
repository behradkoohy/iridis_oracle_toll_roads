from Experiment import Experiment

from DiscretePySwarms import IntOptimizerPSO
from TimestepPriceUtils import evaluate_solution_timestepprice
from PSO_CONFIG import *


class ParticleSwarmTimestepPriceTT(Experiment):
    def __init__(self, db_path, ID, model_name, description, votseed, timeseed):
        super().__init__(db_path, ID, model_name, description, votseed, timeseed)
        self.bound = 1
        self.pricing_dict = {
            -1: lambda x: x - self.bound,
            0: lambda x: x,
            1: lambda x: x + self.bound,
        }
        self.run_exp()

    def run_exp(self):
        car_dist_arrival = self.generate_car_time_distribution()
        car_vot_arrival = self.generate_car_vot_distribution()

        def discrete_activate_funct(x):
            # return x + 1
            if x <= -1:
                return -1
            elif x >= 1:
                return 1
            else:
                return 0

        def objective_function(solution):
            solution = [list(map(discrete_activate_funct, sol)) for sol in solution]
            score = [
                -evaluate_solution_timestepprice(
                    sol,
                    car_dist_arrival,
                    car_vot_arrival,
                    self.pricing_dict,
                    self.timesteps,
                    seq_decisions=True,
                    optimise_for="TravelTime",
                )
                for sol in solution
            ]
            return score

        max_bound = [2 for _ in range((self.timesteps + 1) * 2)]
        min_bound = [-2 for _ in range((self.timesteps + 1) * 2)]
        bounds = (min_bound, max_bound)
        options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
        # optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=100, options=options, bounds=bounds)
        # cost, pos = optimizer.optimize(objective_function, iters=1000)
        optimizer = IntOptimizerPSO(
            n_particles=N_PARTICLES, dimensions=62, options=options, bounds=bounds
        )
        cost, pos = optimizer.optimize(objective_function, iters=N_ITERATIONS)
        pos = list(map(discrete_activate_funct, pos))
        travel_time, social_cost, combined_cost = evaluate_solution_timestepprice(
            pos,
            car_dist_arrival,
            car_vot_arrival,
            self.pricing_dict,
            self.timesteps,
            seq_decisions=True,
            post_eval=True,
            optimise_for="TravelTime",
        )
        self.write_results("TravelTime", *travel_time)
        self.write_results("SocialCost", *social_cost)
        self.write_results("CombinedCost", *combined_cost)
        print(cost, pos)

    # def main(self):


if __name__ == "__main__":
    pso = ParticleSwarmTimestepPriceTT("", 0, 0, 0, 0, 0)

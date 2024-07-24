from Experiment import Experiment
import pyswarms as ps
from TimeOnlyUtils import reduced_evaluate_solution, convert_to_vot_sol


class ParticleSwarmTimeOnly(Experiment):
    def __init__(self, db_path, ID, model_name, description, votseed, timeseed):
        super().__init__(db_path, ID, model_name, description, votseed, timeseed)
        self.run_exp()

    def run_exp(self):
        car_dist_arrival = self.generate_car_time_distribution()
        car_vot_arrival = self.generate_car_vot_distribution()
        def discrete_activate_funct(x):
            return x + 1

        def objective_function(solution):
            solution = [list(map(discrete_activate_funct, sol)) for sol in solution]
            return [-reduced_evaluate_solution(sol, car_dist_arrival, self.timesteps, seq_decisions=True) for sol in solution]

        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 100, 'p': 2}
        optimizer = ps.discrete.BinaryPSO(n_particles=100, dimensions=100, options=options)
        cost, pos = optimizer.optimize(objective_function, iters=1000)
        pos = list(map(discrete_activate_funct, pos))
        self.write_results(
            "TravelTime",
            *reduced_evaluate_solution(pos, car_dist_arrival, self.timesteps, seq_decisions=True, post_eval=True)
        )
        self.write_results(
            "SocialCost",
            *convert_to_vot_sol(pos, car_dist_arrival, car_vot_arrival, self.timesteps, seq_decisions=True, post_eval=True)
        )
        print(cost, pos)


    # def main(self):


if __name__ == '__main__':
    pso = ParticleSwarmTimeOnly("",0,0,0,0,0)


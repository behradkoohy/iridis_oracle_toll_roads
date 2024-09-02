from abc import ABC, abstractmethod
import sqlite3
import numpy.random as nprand


class Experiment:

    def __init__(self, db_path, ID, model_name, description, votseed, timeseed):
        # passed in parameters
        self.db_path = db_path
        self.ID = ID
        self.model_name = model_name
        self.description = description
        self.votseed = votseed
        self.timeseed = timeseed

        # parameters that should mostly stay the same. if they need changing, changing in here should edit all exps
        self.timesteps = 1000
        self.beta_dist_alpha = 5
        self.beta_dist_beta = 5
        self.n_cars = 1000

        self.car_vot_upperbound = 1.0
        self.car_vot_lowerbound = 0.0

        self.db_tables = {"TravelTime", "SocialCost", "CombinedCost"}

    def generate_car_time_distribution(self):

        nprand.seed(self.timeseed)
        car_dist_norm = nprand.beta(
            self.beta_dist_alpha,
            self.beta_dist_beta,
            size=self.n_cars,
        )
        car_dist_arrival = list(
            map(
                lambda z: round(
                    (z - min(car_dist_norm))
                    / (max(car_dist_norm) - min(car_dist_norm))
                    * self.timesteps
                ),
                car_dist_norm,
            )
        )
        return car_dist_arrival

    def generate_car_vot_distribution(self):
        nprand.seed(self.votseed)
        car_vot = nprand.uniform(
            self.car_vot_lowerbound, self.car_vot_upperbound, self.n_cars
        )
        return car_vot

    def check_table_valid(self, prop_table):
        return prop_table in self.db_tables

    def write_results(self, table, min, Q1, med, mean, Q3, max, std, atkidx, ginicoef):
        if not self.check_table_valid(table):
            raise ValueError("results: status must be one of %r." % self.db_tables)
        self.conn = sqlite3.connect(self.db_path)
        self.cur = self.conn.cursor()
        insert_statement = f"""
            INSERT INTO {table} (ID, min, q1, med, mean, q3, max, stdev, atkidx, ginicoef)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.cur.execute(
            insert_statement,
            (self.ID, min, Q1, med, mean, Q3, max, std, atkidx, ginicoef),
        )
        self.conn.commit()

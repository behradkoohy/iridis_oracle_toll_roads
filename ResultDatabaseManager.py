"""
This script should be called by a bash script that sets up jobs for iridis.

Its' primary function is to create an entry in a database that stores the results from the experiments
"""

import sqlite3
from datetime import datetime


class DatabaseCreate:
    def __init__(self, db_path=None):
        if db_path == None:
            now = datetime.now()
            self.db_path = now.strftime("%d%m%y-%H%M%S") + ".db"
        else:
            self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, timeout=2000)
        self.cur = self.conn.cursor()
        # Now we create our tables
        self.cur.execute(
            """
            CREATE TABLE IF NOT EXISTS Models (
                ID INTEGER PRIMARY KEY,
                ModelName TEXT NOT NULL,
                Description TEXT,
                VOTSeed INTEGER,
                TIMESeed INTEGER
            );
        """
        )
        self.cur.execute(
            """
            CREATE TABLE IF NOT EXISTS TravelTime (
                ID INTEGER PRIMARY KEY,
                min REAL,
                q1 REAL,
                med REAL,
                mean REAL,
                q3 REAL,
                max REAL,
                stdev REAL,
                atkidx REAL,
                ginicoef REAL,
                FOREIGN KEY(ID) REFERENCES Models(ID)
            );
        """
        )
        self.cur.execute(
            """
            CREATE TABLE IF NOT EXISTS SocialCost (
                ID INTEGER PRIMARY KEY,
                min REAL,
                q1 REAL,
                med REAL,
                mean REAL,
                q3 REAL,
                max REAL,
                stdev REAL,
                atkidx REAL,
                ginicoef REAL,
                FOREIGN KEY(ID) REFERENCES Models(ID)
            );
        """
        )
        self.cur.execute(
            """
            CREATE TABLE IF NOT EXISTS CombinedCost (
                ID INTEGER PRIMARY KEY,
                min REAL,
                q1 REAL,
                med REAL,
                mean REAL,
                q3 REAL,
                max REAL,
                stdev REAL,
                atkidx REAL,
                ginicoef REAL,
                FOREIGN KEY(ID) REFERENCES Models(ID)
            );
        """
        )
        self.conn.commit()

    def get_db_path(self):
        return self.db_path

    def add_experiment(self, ID, model_name, description, votseed, timeseed):
        self.cur.execute(
            """
            INSERT INTO Models (ID, ModelName, Description, VOTSeed, TIMESeed)
            VALUES (?, ?, ?, ?, ?)
            """,
            (ID, model_name, description, votseed, timeseed),
        )
        self.conn.commit()

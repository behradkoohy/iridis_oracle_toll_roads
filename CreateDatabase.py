import argparse
import sqlite3
from datetime import datetime


class DatabaseInit:
    def __init__(self, db_path=None):
        if db_path == None:
            now = datetime.now()
            self.db_path = now.strftime("%d%m%y-%H%M%S") + ".db"
        else:
            self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.cur = self.conn.cursor()
        # Now we create our tables
        self.cur.execute(
            """
            CREATE TABLE Models (
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
            CREATE TABLE TravelTime (
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
            CREATE TABLE SocialCost (
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Insert a new model into the database.')
    parser.add_argument('db_path', type=str, help='The path to the SQLite database')
    args = parser.parse_args()
    db = DatabaseInit(args.db_path)
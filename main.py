from Experiment import Experiment
from PSOFixedPriceTTOpt import ParticleSwarmFixedPriceTT
from PSOFixedPriceSCOpt import ParticleSwarmFixedPriceSC
from PSOFixedPriceCCOpt import ParticleSwarmFixedPriceCC
from PSOFTimestepPriceTTOpt import ParticleSwarmTimestepPriceTT
from PSOFTimestepPriceSCOpt import ParticleSwarmTimestepPriceSC
from PSOFTimestepPriceCCOpt import ParticleSwarmTimestepPriceCC
from PSOFUnboundPriceTTOpt import ParticleSwarmUnboundPriceTT
from PSOFUnboundPriceSCOpt import ParticleSwarmUnboundPriceSC
from PSOFUnboundPriceCCOpt import ParticleSwarmUnboundPriceCC
from PSOLinearPriceTTOpt import ParticleSwarmLinearPriceTT
from PSOLinearPriceSCOpt import ParticleSwarmLinearPriceSC
from PSOLinearPriceCCOpt import ParticleSwarmLinearPriceCC
from PSOTimeOnly import ParticleSwarmTimeOnly
from ResultDatabaseManager import DatabaseCreate
import argparse
import sqlite3

# model_dict = {
#     "PSOTimeOnly": ParticleSwarmTimeOnly,
#     "PSOTimestepPriceSCOpt": ParticleSwarmTimestepPriceSC,
#     "PSOFTimestepPriceTTOpt": ParticleSwarmTimestepPriceTT,
#     "PSOFTimestepPriceCCOpt": ParticleSwarmTimestepPriceCC,
#
# }
model_dict = {
    "PSOTimeOnly": ParticleSwarmTimeOnly,
    "PSOFTimestepPriceSCOpt": ParticleSwarmTimestepPriceSC,
    "PSOFTimestepPriceTTOpt": ParticleSwarmTimestepPriceTT,
    "PSOFTimestepPriceCCOpt": ParticleSwarmTimestepPriceCC,
    "PSOFixedPriceTTOpt": ParticleSwarmFixedPriceTT,
    "PSOFixedPriceSCOpt": ParticleSwarmFixedPriceSC,
    "PSOFixedPriceCCOpt": ParticleSwarmFixedPriceCC,
    "PSOFUnboundPriceTTOpt": ParticleSwarmUnboundPriceTT,
    "PSOFUnboundPriceSCOpt": ParticleSwarmUnboundPriceSC,
    "PSOFUnboundPriceCCOpt": ParticleSwarmUnboundPriceCC,
    "PSOLinearPriceTTOpt": ParticleSwarmLinearPriceTT,
    "PSOLinearPriceSCOpt": ParticleSwarmLinearPriceSC,
    "PSOLinearPriceCCOpt": ParticleSwarmLinearPriceCC,
}


class DummyExp(Experiment):
    def __init__(self, db_path, ID, model_name, description, votseed, timeseed):
        super().__init__(db_path, ID, model_name, description, votseed, timeseed)
        self.run_exp()

    def run_exp(self):
        print(self.generate_car_time_distribution())
        print(self.generate_car_vot_distribution())
        self.write_results("SocialCost", 1, 2, 3, 4, 5, 6, 7, 8, 9)


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    """
    Don't run a loop in here, this just runs ONE experiment.
    """
    # Create the database that we're working with
    parser = argparse.ArgumentParser(
        description="Insert a new model into the database."
    )
    parser.add_argument("db_path", type=str, help="The path to the SQLite database")
    parser.add_argument("ID", type=int, help="The ID of the model")
    parser.add_argument("ModelName", type=str, help="The name of the model")
    parser.add_argument(
        "Description", type=str, help="A short description of the model"
    )
    parser.add_argument("VOTSeed", type=int, help="The VOT seed value")
    parser.add_argument("TIMESeed", type=int, help="The TIME seed value")

    # Parse the command line arguments
    args = parser.parse_args()
    db_manager = DatabaseCreate(db_path=args.db_path)
    # Verify the ModelName is in the dictionary and get the corresponding class
    model_class = model_dict.get(args.ModelName)
    if not model_class:
        print(f"Error: ModelName '{args.ModelName}' not recognized.")
        exit()

    # Prepare data tuple
    data = (args.ID, args.ModelName, args.Description, args.VOTSeed, args.TIMESeed)
    db_manager.add_experiment(*data)

    # run the experiment
    experiment = model_dict[args.ModelName](args.db_path, *data)

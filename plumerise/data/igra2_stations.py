from importlib.resources import files

import pandas as pd

with files(__package__).joinpath("igra2-station-list.pkl").open("rb") as xlsstream:
    StationList = pd.read_pickle(xlsstream)

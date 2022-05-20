import pandas as pd
from pkg_resources import resource_stream

xlsstream = resource_stream(__name__, "../data/igra2-station-list.pkl")

StationList = pd.read_pickle(xlsstream)

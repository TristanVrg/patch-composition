import pandas as pd
import speasy as spz
from datetime import datetime, timedelta

""" Get Data """

# --- Parker Solar Probe
def get_ephemeris_psp(start, stop):
    start_extended, stop_extended = start - timedelta(hours=1), stop + timedelta(hours=1)
    psp_r_sun = spz.get_data("amda/psp_r_sun", start_extended, stop_extended).to_dataframe()
    car_lon = spz.get_data("amda/psp_lon_sun", start_extended, stop_extended).to_dataframe()
    car_lat = spz.get_data("amda/psp_lat_sun", start_extended, stop_extended).to_dataframe()
    return psp_r_sun, car_lon, car_lat


# --- Solar Orbiter
def get_data_solo(start, stop):
    return


""" Utils """

def average(data, window): 
    data = pd.Series(data)
    avg = data.rolling(window=window, center=True).mean()
    return avg


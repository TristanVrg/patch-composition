""" Import modules """

import warnings

import numpy  as np
import pandas as pd
import xarray as xr
import speasy as spz

from pathlib  import Path
from datetime import datetime, timedelta


""" Constants """

r0    = 696_342_000      # Sun's Radius [m]
omega = 2.9e-6           # Sun’s angular moment taken at the equator [s-1]
MU0   = 4 * np.pi * 1e-7 # Vacuum permeability [N.A-2]
KB    = 1.380649e-23     # Boltzman constant [J.K-1]
mp    = 1.67262192e-27   # Proton mass [kg]
e     = 1.602176634e-19  # Elementary charge [A.s]


""" Get Data """

# --- Parker Solar Probe
def _load_data_spi_psp(start: datetime, stop: datetime) -> pd.DataFrame:
    """Load Parker Solar Probe span-i proton data."""

    np_ = (
        spz.get_data("amda/psp_spi_Hn", start, stop)
        .to_dataframe()
        .iloc[:, 0]
    )
    
    na = (
        spz.get_data("amda/psp_spi_an", start, stop)
        .to_dataframe()
        .iloc[:,0]
    )
    
    vp_rtn = (
        spz.get_data("amda/psp_spi_Hv", start, stop)
        .to_dataframe()
    )

    tp = (
        spz.get_data("amda/psp_spi_Hw", start, stop)
        .to_dataframe()
        .iloc[:, 0]
    )

    return pd.DataFrame({
        "Np": np_,
        "Na": na,
        "Vp_r": vp_rtn.iloc[:, 0],
        "Vp_t": vp_rtn.iloc[:, 1],
        "Vp_n": vp_rtn.iloc[:, 2],
        "Tp": tp,
    })


def _load_data_mag_psp(start: datetime, stop: datetime) -> pd.DataFrame:
    """Load Parker Solar Probe MAG data."""

    b_rtn = (
        spz.get_data("amda/psp_b_4cyc", start, stop)
        .to_dataframe()
    )

    return pd.DataFrame({
        "Br": b_rtn.iloc[:, 0],
        "Bt": b_rtn.iloc[:, 1],
        "Bn": b_rtn.iloc[:, 2],
    })


def _load_data_ephemeris_psp(
    start: datetime,
    stop: datetime,
    delta: float = 1,
) -> pd.DataFrame:
    """Load Parker Solar Probe ephemeris."""

    start_extended = start - timedelta(hours=delta)
    stop_extended = stop + timedelta(hours=delta)

    r_psp_sun = (
        spz.get_data(
            "amda/psp_r_sun",
            start_extended,
            stop_extended,
        )
        .to_dataframe()
        .iloc[:, 0]
    )

    car_lon = (
        spz.get_data(
            "amda/psp_lon_sun",
            start_extended,
            stop_extended,
        )
        .to_dataframe()
        .iloc[:, 0]
    )

    car_lat = (
        spz.get_data(
            "amda/psp_lat_sun",
            start_extended,
            stop_extended,
        )
        .to_dataframe()
        .iloc[:, 0]
    )


    return pd.DataFrame({
        "car_lon": car_lon,
        "car_lat": car_lat,
        "r_sun": r_psp_sun,
    })


def load_data_psp(
    start: datetime,
    stop: datetime,
) -> pd.DataFrame:

    data_spi = _load_data_spi_psp(start, stop)
    data_mag = _load_data_mag_psp(start, stop)
    data_eph = _load_data_ephemeris_psp(start, stop)

    data_spi = data_spi.sort_index()
    data_mag = data_mag.sort_index()
    data_eph = data_eph.sort_index()

    data_spi = data_spi[~data_spi.index.duplicated()]
    data_mag = data_mag[~data_mag.index.duplicated()]
    data_eph = data_eph[~data_eph.index.duplicated()]

    mag_res = resample_dataframe(data_mag, data_spi.index)
    eph_res = resample_dataframe(data_eph, data_spi.index)

    return pd.concat(
        [eph_res, data_spi, mag_res],
        axis=1,
    )


# --- Solar Orbiter
def _read_velocirap_file(start, stop):
    
    datelist = create_datetime_list(start, stop)
    
    file_list = []
    
    # --- Load all files
    for (start_temp, stop_temp) in datelist:
        
        filepath = Path('..') / f"Data/velocirap/SWA-PAS-MOM_{start_temp}_{stop_temp}_1s3p.nc"
        
        data_temp = xr.open_dataset(filepath).to_dataframe()
        file_list.append(data_temp)
    
    # --- Combine all files
    data = pd.concat(file_list)
    
    # --- Keep only the requested time interval
    data = data.loc[start:stop]
    
    return data

    
def _load_data_pas_solo(start: datetime, stop: datetime) -> pd.DataFrame:
    """Load Solar Orbiter PAS data from AMDA."""

    np_ = (
        spz.get_data("amda/pas_momgr_n", start, stop)
        .to_dataframe()
        .iloc[:, 0]
    )

    vp_rtn = (
        spz.get_data("amda/pas_momgr1_v_rtn", start, stop)
        .to_dataframe()
    )

    Tp_rtn = (
        spz.get_data("amda/pas_momgr_trtn", start, stop)
        .to_dataframe()
    )

    Tp = Tp_rtn.mean(axis=1)

    return pd.DataFrame({
        "Np": np_,
        "Vp_r": vp_rtn.iloc[:, 0],
        "Vp_t": vp_rtn.iloc[:, 1],
        "Vp_n": vp_rtn.iloc[:, 2],
        "Tp": Tp,
    })


def _load_data_mag_solo(start: datetime, stop: datetime) -> pd.DataFrame:
    """Load Solar Orbiter MAG data."""

    mag_key = (
        spz.inventories.tree.cda
        .Solar_Orbiter.SOLO.MAG.SOLO_L2_MAG_RTN_NORMAL.B_RTN
    )

    b_rtn = (
        spz.cda.get_data(mag_key, start, stop)
        .to_dataframe()
    )

    return pd.DataFrame({
        "Br": b_rtn.iloc[:, 0],
        "Bt": b_rtn.iloc[:, 1],
        "Bn": b_rtn.iloc[:, 2],
    })


def _load_data_ephemeris_solo(
    start: datetime,
    stop: datetime,
    delta: float = 1,
) -> pd.DataFrame:
    """Load Solar Orbiter ephemeris."""

    start_extended = start - timedelta(hours=delta)
    stop_extended = stop + timedelta(hours=delta)

    r_so_sun = (
        spz.get_data(
            "amda/so_r_sun",
            start_extended,
            stop_extended,
        )
        .to_dataframe()
        .iloc[:, 0]
    )

    car_lon = (
        spz.get_data(
            "amda/so_lon_sun",
            start_extended,
            stop_extended,
        )
        .to_dataframe()
        .iloc[:, 0]
    )

    car_lat = (
        spz.get_data(
            "amda/so_lat_sun",
            start_extended,
            stop_extended,
        )
        .to_dataframe()
        .iloc[:, 0]
    )

    return pd.DataFrame({
        "car_lon": car_lon,
        "car_lat": car_lat,
        "r_sun": r_so_sun,
    })


def load_data_solo(
    start: datetime,
    stop: datetime,
    velocirap: bool = False
) -> pd.DataFrame:
    
    data_pas = (
        _read_velocirap_file(start, stop)
        if velocirap
        else _load_data_pas_solo(start, stop)
    )

    data_mag = _load_data_mag_solo(start, stop)
    data_eph = _load_data_ephemeris_solo(start, stop)

    data_pas = data_pas.sort_index()
    data_mag = data_mag.sort_index()
    data_eph = data_eph.sort_index()

    data_pas = data_pas[~data_pas.index.duplicated()]
    data_mag = data_mag[~data_mag.index.duplicated()]
    data_eph = data_eph[~data_eph.index.duplicated()]

    mag_res = resample_dataframe(data_mag, data_pas.index)
    eph_res = resample_dataframe(data_eph, data_pas.index)

    return pd.concat(
        [eph_res, data_pas, mag_res],
        axis=1,
    )


def load_data(
    start: datetime,
    stop: datetime,
    spacecraft: str,
    **kwargs,
) -> pd.DataFrame:
    """Load spacecraft data."""

    spacecraft = spacecraft.lower()

    loaders = {
        "solo": load_data_solo,
        "psp": load_data_psp,
    }

    if spacecraft not in loaders:
        warnings.warn(
            f'Unknown spacecraft "{spacecraft}". '
            "Assuming Solar Orbiter.",
            UserWarning,
            stacklevel=2,
        )
        spacecraft = "solo"

    data = loaders[spacecraft](start, stop, **kwargs)

    data.attrs["spacecraft"] = spacecraft

    return data


""" Normalised Deflection """

def _average(time_start, time_end, spacecraft, duration, ref_index):

    interval = timedelta(seconds=duration/2)

    start_extend = time_start - interval
    stop_extend  = time_end  + interval

    loaders = {
        "solo": "amda/pas_momgr1_v_rtn",
        "psp": "amda/psp_spi_Hv"
    }

    data = spz.get_data(
        loaders[spacecraft],
        start_extend,
        stop_extend
    ).to_dataframe()['vr']

    data = data.sort_index()
    data = data[~data.index.duplicated()]

    data_mean = data.rolling(
        f"{duration}s",
        center=True
    ).mean()

    data = data.reindex(ref_index)
    data_mean = data_mean.reindex(ref_index)

    return data, data_mean


def _sliding_window_r(r, n):
    """
    Apply a centered sliding-window criterion to normalized deflection z.

    Parameters
    ----------
    r : array-like
        Radial magnetic field time series.
    n : int
        Number of points considered on each side of the central point.

    Returns
    -------
    result : numpy.ndarray
        1 if the mean z in the window is > 0.5, 0 otherwise.
    """

    r = np.asarray(r)
    result = np.zeros(len(r), dtype=int)

    for i in range(len(r)):

        # Window boundaries
        start = max(0, i - n)
        stop  = min(len(r), i + n + 1)

        # Mean z in the window
        mean_r = np.nanmean(r[start:stop])

        # Classification
        result[i] = 0 if mean_r > 0 else 1

    return result


def normalised_deflection(data):
    
    spacecraft = data.attrs.get("spacecraft")
    start = data.index[0].round('D').to_pydatetime()
    stop  = data.index[-1].round('D').to_pydatetime()
    
    r_sun = data['r_sun']
    br, bt, bn = data['Br'], data['Bt'], data['Bn']
    b_mag = np.sqrt(br**2 + bt**2 + bn**2)
    
    K = _sliding_window_r(br, 5000)

    # Reconstructing Parker Spiral Angle       
    r = r_sun - r0  

    vr, vr_mean = _average(start, stop, spacecraft, 3600, data.index)
    phi_p = np.arctan(np.radians(-(omega*r)/vr_mean)) + K*np.pi
    
    # Reconstructing Parker Spiral Field            
    Bp_r = b_mag * np.cos(phi_p)
    Bp_t = b_mag * np.sin(phi_p)
    
    # Computing delflection                
    Bt_inverse = 1/(b_mag**2)
    Bp_Bt = Bp_r*br + Bp_t*bt
    cos_alpha = Bt_inverse * Bp_Bt
    
    z = 0.5*(1-cos_alpha)

    return z


""" Utils """

def create_datetime_list(start: datetime, stop: datetime):
    datetime_list = []

    current = start.replace(hour=0, minute=0, second=0, microsecond=0)

    while current < stop:
        next_day = current + timedelta(days=1)

        # Ne pas dépasser la date de fin
        # next_day = min(next_day, stop)

        datetime_list.append((current, next_day))

        current = next_day

    return datetime_list


def compute_interval_parameters(
    data: pd.DataFrame,
    start: datetime,
    stop: datetime,
) -> dict:
    """
    Compute derived parameters for a catalog interval.
    """

    duration = stop - start

    r_min = data["r_sun"].min()
    r_max = data["r_sun"].max()

    return {
        "date_start": start,
        "date_end": stop,
        "duration": duration.total_seconds(),
        "r_min": r_min,
        "r_max": r_max,
    }


def build_catalog(catalog_inputs: list[dict]) -> pd.DataFrame:

    DATA_LOADERS = {
        "Solar Orbiter": load_data_solo,
        "Parker Solar Probe": load_data_psp,
    }
    
    spacecraft_names = {
        "Solar Orbiter": "SOLO",
        "Parker Solar Probe": "PSP",
    }
    
    catalog = []

    for catalog_input in catalog_inputs:

        n_intervals = len(catalog_input["spacecraft"])

        for i in range(n_intervals):

            spacecraft = catalog_input["spacecraft"][i]
            start = catalog_input["date_start"][i]
            stop = catalog_input["date_end"][i]

            print(
                f"{spacecraft}: "
                f"{start} -> {stop}"
            )

            if spacecraft not in DATA_LOADERS:
                raise ValueError(
                    f"Unknown spacecraft: {spacecraft}"
                )

            # Load
            data = DATA_LOADERS[spacecraft](start, stop)

            # Derived parameters
            parameters = compute_interval_parameters(
                data,
                start,
                stop,
            )

            parameters["spacecraft"] = spacecraft

            catalog.append(parameters)

    catalog = pd.DataFrame(catalog)

    catalog = catalog[
        [
            "spacecraft",
            "date_start",
            "date_end",
            "duration",
            "r_min",
            "r_max",
        ]
    ]

    catalog = catalog.sort_values(
        ["spacecraft", "r_min"]
    ).reset_index(drop=True)
    
    catalog["number"] = (
        catalog.groupby("spacecraft").cumcount() + 1)
    
    catalog["id"] = (
        catalog["spacecraft"].map(spacecraft_names)
        + "-"
        + catalog["number"].astype(str).str.zfill(3))
    
    return catalog


def save_catalog_netcdf(
    catalog: pd.DataFrame,
    filepath: str,
):
    """
    Save catalog DataFrame to NetCDF.
    """

    ds = catalog.to_xarray()

    ds.to_netcdf(filepath)

    print(f"Catalog saved to: {filepath}")


def df_to_latex(df, caption=None, label=None):
    latex = df.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label=label,
        position="htbp"
    )

    latex = latex.replace(
        r"\begin{tabular}",
        r"\centering" + "\n" + r"\begin{tabular}"
    )

    return latex


def resample_dataframe(data: pd.DataFrame, new_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Resample dataframe to a new time index using time interpolation"""

    data = data.copy()

    # garantir ordre temporel
    data = data.sort_index()

    # supprimer doublons temporels
    data = data[~data.index.duplicated(keep="first")]

    # interpolation
    data_interp = (
        data
        .reindex(data.index.union(new_index))
        .interpolate(method="time")
    )

    return data_interp.loc[new_index]


def average(data, window):
    data = pd.Series(data)
    avg = data.rolling(
        window=window,
        center=True,
        min_periods=1
    ).mean()
    return avg


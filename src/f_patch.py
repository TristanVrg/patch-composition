import pandas as pd
import speasy as spz
from datetime import datetime, timedelta

""" Get Data """

# --- Parker Solar Probe
def _load_data_pas_psp(start: datetime, stop: datetime) -> pd.DataFrame:
    """Load Parker Solar Probe proton data."""

    np_ = (
        spz.get_data(
            "amda/psp_spi_Hn",
            start,
            stop,
        )
        .to_dataframe()
        .iloc[:, 0]
    )

    vp_rtn = (
        spz.get_data(
            "amda/psp_spi_Hv",
            start,
            stop,
        )
        .to_dataframe()
    )

    tp = (
        spz.get_data(
            "amda/psp_spi_Hw",
            start,
            stop,
        )
        .to_dataframe()
        .iloc[:, 0]
    )

    return pd.DataFrame({
        "Np": np_,
        "Vp_r": vp_rtn.iloc[:, 0],
        "Vp_t": vp_rtn.iloc[:, 1],
        "Vp_n": vp_rtn.iloc[:, 2],
        "Tp": tp,
    })


def _load_data_mag_psp(start: datetime, stop: datetime) -> pd.DataFrame:
    """Load Parker Solar Probe MAG data."""

    b_rtn = (
        spz.get_data(
            "amda/psp_b_4cyc",
            start,
            stop,
        )
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

    data_spi = _load_data_pas_psp(start, stop)
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
    velocirap: bool = False,
    filepath=None,
) -> pd.DataFrame:

    # data_pas = (
    #     _read_velocirap_file(filepath)
    #     if velocirap
    #     else _load_data_pas_solo(start, stop)
    # )

    data_pas = _load_data_pas_solo(start, stop)
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


""" Utils """

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
    avg = data.rolling(window=window, center=True).mean()
    return avg


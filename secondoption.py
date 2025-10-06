import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

import numpy as np
import xarray as xr
from dateutil import parser as dtparser
from tqdm import tqdm

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, BigInteger,
    ForeignKey, UniqueConstraint, Index, Text, Boolean
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.dialects.postgresql import JSONB

# ---------- Config ----------
PGURL = os.getenv("PGURL", "postgresql+psycopg2://postgres:postgres@localhost:5432/argo")
NETCDF_DIR = Path(os.getenv("ARGO_NETCDF_DIR", "./data")).resolve()
INVENTORY_DIR = Path(os.getenv("ARGO_INVENTORY_DIR", "./data/inventories")).resolve()

# Filenames (adjust if yours differ)
META_NC   = NETCDF_DIR / "1900121_meta.nc"
PROF_NC   = NETCDF_DIR / "1900121_prof.nc"
RTRAJ_NC  = NETCDF_DIR / "1900121_Rtraj.nc"
TECH_NC   = NETCDF_DIR / "1900121_tech.nc"

# ---------- SQLAlchemy Base ----------
Base = declarative_base()

# ---------- ORM Models ----------
class FloatMeta(Base):
    __tablename__ = "floats_meta"
    # WMO platform number is the natural key here
    platform_number = Column(String, primary_key=True)

    # Common admin
    data_centre = Column(String)
    project_name = Column(String)
    pi_name = Column(String)

    # Hardware
    platform_family = Column(String)
    platform_type = Column(String)
    platform_maker = Column(String)
    firmware_version = Column(String)
    manual_version = Column(String)
    float_serial_no = Column(String)
    wmo_inst_type = Column(String)
    battery_type = Column(String)
    battery_packs = Column(String)

    # Deployment
    launch_date = Column(DateTime)            # from LAUNCH_DATE
    launch_latitude = Column(Float)
    launch_longitude = Column(Float)
    launch_qc = Column(String)
    end_mission_date = Column(DateTime)
    end_mission_status = Column(String)
    deployment_platform = Column(String)
    deployment_cruise_id = Column(String)

    # Comms/positioning
    trans_system = Column(String)
    trans_system_id = Column(String)
    trans_frequency = Column(String)
    positioning_system = Column(String)
    ptt = Column(String)

    # Configs (JSON mappings)
    launch_config_params = Column(JSONB)      # {name: value}
    config_params = Column(JSONB)            # {name: [values...] or name:value}
    config_mission_number = Column(Integer)
    config_mission_comment = Column(Text)

    # Arrays of sensors with calib & units
    sensors = relationship("FloatSensor", back_populates="float_meta", cascade="all, delete-orphan")

    # indexing helpers
    __table_args__ = (
        Index("ix_floats_meta_project_name", "project_name"),
        Index("ix_floats_meta_launch_date", "launch_date"),
    )


class FloatSensor(Base):
    __tablename__ = "float_sensors"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    platform_number = Column(String, ForeignKey("floats_meta.platform_number", ondelete="CASCADE"))

    sensor = Column(String)          # e.g., "CTD"
    sensor_maker = Column(String)
    sensor_model = Column(String)
    sensor_serial_no = Column(String)

    parameter = Column(String)       # e.g., "PRES" / "TEMP" / "PSAL"
    parameter_units = Column(String)
    parameter_accuracy = Column(String)
    parameter_resolution = Column(String)

    predeploy_calib_equation = Column(Text)
    predeploy_calib_coefficient = Column(Text)
    predeploy_calib_comment = Column(Text)

    float_meta = relationship("FloatMeta", back_populates="sensors")
    __table_args__ = (
        Index("ix_float_sensors_platform_number", "platform_number"),
    )


class Profile(Base):
    __tablename__ = "profiles"
    id = Column(BigInteger, primary_key=True, autoincrement=True)

    platform_number = Column(String, ForeignKey("floats_meta.platform_number", ondelete="CASCADE"), index=True)
    cycle_number = Column(Integer, index=True)
    juld = Column(Float)  # Julian days since reference
    timestamp = Column(DateTime, index=True)

    latitude = Column(Float)
    longitude = Column(Float)

    direction = Column(String)
    data_centre = Column(String)
    data_state_indicator = Column(String)
    data_mode = Column(String)
    positioning_system = Column(String)
    config_mission_number = Column(Integer)

    # QC on whole profile
    juld_qc = Column(String)
    position_qc = Column(String)
    profile_pres_qc = Column(String)
    profile_temp_qc = Column(String)
    profile_psal_qc = Column(String)

    # calib blobs (per-parameter)
    scientific_calib = Column(JSONB)  # [{"parameter":"TEMP","equation":"...","coeff":"...","comment":"...","date":"..."}]

    measurements = relationship("ProfileMeasurement", back_populates="profile", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("platform_number", "cycle_number", name="uq_profile_float_cycle"),
        Index("ix_profiles_lat_lon", "latitude", "longitude"),
    )


class ProfileMeasurement(Base):
    __tablename__ = "profile_measurements"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    profile_id = Column(BigInteger, ForeignKey("profiles.id", ondelete="CASCADE"), index=True)
    depth_index = Column(Integer)  # 0..N_LEVELS-1

    pres = Column(Float)
    temp = Column(Float)
    psal = Column(Float)

    pres_adjusted = Column(Float)
    temp_adjusted = Column(Float)
    psal_adjusted = Column(Float)

    pres_qc = Column(String)
    temp_qc = Column(String)
    psal_qc = Column(String)

    pres_adjusted_error = Column(Float)
    temp_adjusted_error = Column(Float)
    psal_adjusted_error = Column(Float)

    profile = relationship("Profile", back_populates="measurements")
    __table_args__ = (
        Index("ix_prof_meas_profile_depth", "profile_id", "depth_index"),
    )


class TrajectoryPoint(Base):
    __tablename__ = "trajectory_points"
    id = Column(BigInteger, primary_key=True, autoincrement=True)

    platform_number = Column(String, ForeignKey("floats_meta.platform_number", ondelete="CASCADE"), index=True)
    cycle_number = Column(Integer, index=True)

    juld = Column(Float)
    timestamp = Column(DateTime, index=True)

    latitude = Column(Float)
    longitude = Column(Float)

    measurement_code = Column(Integer)  # MEASUREMENT_CODE
    position_accuracy = Column(String)
    position_qc = Column(String)

    pres = Column(Float)
    temp = Column(Float)
    psal = Column(Float)

    pres_adjusted = Column(Float)
    temp_adjusted = Column(Float)
    psal_adjusted = Column(Float)

    pres_adjusted_error = Column(Float)
    temp_adjusted_error = Column(Float)
    psal_adjusted_error = Column(Float)

    data_mode = Column(String)  # if available at point level


class TrajectoryEvent(Base):
    __tablename__ = "trajectory_events"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    platform_number = Column(String, ForeignKey("floats_meta.platform_number", ondelete="CASCADE"), index=True)
    cycle_number = Column(Integer, index=True)

    event_name = Column(String)       # e.g., "ASCENT_START", "DESCENT_END"
    juld = Column(Float)
    timestamp = Column(DateTime)
    status = Column(String)           # *_STATUS if present


class TechnicalParameter(Base):
    __tablename__ = "technical_parameters"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    platform_number = Column(String, ForeignKey("floats_meta.platform_number", ondelete="CASCADE"), index=True)
    cycle_number = Column(Integer, index=True)
    parameter_name = Column(String, index=True)
    parameter_value = Column(Text)    # keep as text; parse to numeric later if needed


# ---------- Engine / Session ----------
engine = create_engine(PGURL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine)

# ---------- Helpers ----------
def safe_get(ds: xr.Dataset, key: str) -> Optional[Any]:
    if key in ds.variables:
        return ds[key].values
    if key in ds.attrs:
        return ds.attrs[key]
    return None

def ensure_py(obj):
    """Convert numpy types to Python scalars/lists for JSONB."""
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def parse_ref_datetime(ds: xr.Dataset) -> datetime:
    """
    ARGO commonly uses REFERENCE_DATE_TIME = '19500101000000'.
    If missing/invalid, default to 1950-01-01T00:00:00Z.
    """
    ref = safe_get(ds, "REFERENCE_DATE_TIME")
    if ref is None:
        return datetime(1950, 1, 1, 0, 0, 0)
    # Could be bytes/bytes array/object array
    if isinstance(ref, (bytes, bytearray)):
        ref = ref.decode("utf-8", "ignore")
    if isinstance(ref, np.ndarray):
        ref = ref.item() if ref.size == 1 else str(ref)
    ref = str(ref)
    # Expect 'YYYYMMDDhhmmss'
    try:
        return datetime.strptime(ref, "%Y%m%d%H%M%S")
    except Exception:
        # Try generic parser
        try:
            return dtparser.parse(ref)
        except Exception:
            return datetime(1950, 1, 1, 0, 0, 0)

def juld_to_timestamp(ref_dt: datetime, juld_val: Any) -> Optional[datetime]:
    try:
        f = float(juld_val)
        if np.isnan(f):
            return None
        return ref_dt + timedelta(days=f)
    except Exception:
        return None

def array_or_none(ds: xr.Dataset, key: str) -> Optional[np.ndarray]:
    try:
        if key in ds.variables:
            vals = ds[key].values
            # xarray might give masked arrays; convert to ndarray with NaNs
            return np.array(vals, dtype=object if vals.dtype == object else None)
        return None
    except Exception:
        return None

def to_str_or_none(x) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", "ignore")
    if isinstance(x, np.ndarray):
        # object/char arrays -> join
        if x.dtype.kind in ("S", "U", "O") and x.ndim > 0:
            # flatten to single string list joined by space
            try:
                return " ".join([to_str_or_none(v) or "" for v in x.tolist()]).strip()
            except Exception:
                return str(x)
        return str(x.tolist())
    return str(x)

# ---------- Ingestors ----------
def ingest_meta(session, meta_nc: Path):
    ds = xr.open_dataset(meta_nc, decode_cf=True, decode_times=False)

    platform_number = to_str_or_none(safe_get(ds, "PLATFORM_NUMBER"))
    if not platform_number:
        raise ValueError("PLATFORM_NUMBER missing in meta file")

    fm = session.get(FloatMeta, platform_number)
    if fm is None:
        fm = FloatMeta(platform_number=platform_number)

    # Basic
    fm.data_centre   = to_str_or_none(safe_get(ds, "DATA_CENTRE"))
    fm.project_name  = to_str_or_none(safe_get(ds, "PROJECT_NAME"))
    fm.pi_name       = to_str_or_none(safe_get(ds, "PI_NAME"))

    # Hardware
    fm.platform_family  = to_str_or_none(safe_get(ds, "PLATFORM_FAMILY"))
    fm.platform_type    = to_str_or_none(safe_get(ds, "PLATFORM_TYPE"))
    fm.platform_maker   = to_str_or_none(safe_get(ds, "PLATFORM_MAKER"))
    fm.firmware_version = to_str_or_none(safe_get(ds, "FIRMWARE_VERSION"))
    fm.manual_version   = to_str_or_none(safe_get(ds, "MANUAL_VERSION"))
    fm.float_serial_no  = to_str_or_none(safe_get(ds, "FLOAT_SERIAL_NO"))
    fm.wmo_inst_type    = to_str_or_none(safe_get(ds, "WMO_INST_TYPE"))
    fm.battery_type     = to_str_or_none(safe_get(ds, "BATTERY_TYPE"))
    fm.battery_packs    = to_str_or_none(safe_get(ds, "BATTERY_PACKS"))

    # Deployment
    fm.launch_date       = juld_to_ts_from_calendar_str(to_str_or_none(safe_get(ds, "LAUNCH_DATE")))
    fm.launch_latitude   = ensure_float(safe_get(ds, "LAUNCH_LATITUDE"))
    fm.launch_longitude  = ensure_float(safe_get(ds, "LAUNCH_LONGITUDE"))
    fm.launch_qc         = to_str_or_none(safe_get(ds, "LAUNCH_QC"))
    fm.end_mission_date  = juld_to_ts_from_calendar_str(to_str_or_none(safe_get(ds, "END_MISSION_DATE")))
    fm.end_mission_status = to_str_or_none(safe_get(ds, "END_MISSION_STATUS"))
    fm.deployment_platform = to_str_or_none(safe_get(ds, "DEPLOYMENT_PLATFORM"))
    fm.deployment_cruise_id = to_str_or_none(safe_get(ds, "DEPLOYMENT_CRUISE_ID"))

    # Comms/positioning
    fm.trans_system      = to_str_or_none(safe_get(ds, "TRANS_SYSTEM"))
    fm.trans_system_id   = to_str_or_none(safe_get(ds, "TRANS_SYSTEM_ID"))
    fm.trans_frequency   = to_str_or_none(safe_get(ds, "TRANS_FREQUENCY"))
    fm.positioning_system = to_str_or_none(safe_get(ds, "POSITIONING_SYSTEM"))
    fm.ptt               = to_str_or_none(safe_get(ds, "PTT"))

    # Launch config pairs
    lnames = array_or_none(ds, "LAUNCH_CONFIG_PARAMETER_NAME")
    lvals  = array_or_none(ds, "LAUNCH_CONFIG_PARAMETER_VALUE")
    fm.launch_config_params = (
        { to_str_or_none(k): ensure_number(lvals[i]) for i, k in enumerate(lnames) }
        if (lnames is not None and lvals is not None) else None
    )

    # Config params (may be 2D [1, N] or [N])
    cnames = array_or_none(ds, "CONFIG_PARAMETER_NAME")
    cvals  = array_or_none(ds, "CONFIG_PARAMETER_VALUE")
    fm.config_params = build_config_params(cnames, cvals)

    cmn = safe_get(ds, "CONFIG_MISSION_NUMBER")
    fm.config_mission_number = int(np.array(cmn).item()) if cmn is not None else None
    fm.config_mission_comment = to_str_or_none(safe_get(ds, "CONFIG_MISSION_COMMENT"))

    # Sensors (parallel 1-D arrays of length ~3)
    sensors = []
    for i in range(max_len_arrays(ds, ["SENSOR", "SENSOR_MAKER", "SENSOR_MODEL", "SENSOR_SERIAL_NO",
                                       "PARAMETER", "PARAMETER_UNITS", "PARAMETER_ACCURACY", "PARAMETER_RESOLUTION",
                                       "PREDEPLOYMENT_CALIB_EQUATION", "PREDEPLOYMENT_CALIB_COEFFICIENT",
                                       "PREDEPLOYMENT_CALIB_COMMENT"]))):
        sensors.append(FloatSensor(
            platform_number=platform_number,
            sensor=pick_idx(ds, "SENSOR", i),
            sensor_maker=pick_idx(ds, "SENSOR_MAKER", i),
            sensor_model=pick_idx(ds, "SENSOR_MODEL", i),
            sensor_serial_no=pick_idx(ds, "SENSOR_SERIAL_NO", i),
            parameter=pick_idx(ds, "PARAMETER", i),
            parameter_units=pick_idx(ds, "PARAMETER_UNITS", i),
            parameter_accuracy=pick_idx(ds, "PARAMETER_ACCURACY", i),
            parameter_resolution=pick_idx(ds, "PARAMETER_RESOLUTION", i),
            predeploy_calib_equation=pick_idx(ds, "PREDEPLOYMENT_CALIB_EQUATION", i),
            predeploy_calib_coefficient=pick_idx(ds, "PREDEPLOYMENT_CALIB_COEFFICIENT", i),
            predeploy_calib_comment=pick_idx(ds, "PREDEPLOYMENT_CALIB_COMMENT", i),
        ))

    # replace children
    fm.sensors.clear()
    fm.sensors.extend([s for s in sensors if s.parameter or s.sensor])

    session.merge(fm)
    session.commit()
    ds.close()
    return platform_number


def ingest_profiles(session, platform_number: str, prof_nc: Path):
    ds = xr.open_dataset(prof_nc, decode_cf=True, decode_times=False)

    ref_dt = parse_ref_datetime(ds)

    # Identify sizes
    # Expect dims: N_PROF x N_LEVELS
    pres = array_or_none(ds, "PRES")
    temp = array_or_none(ds, "TEMP")
    psal = array_or_none(ds, "PSAL")

    if pres is None or temp is None or psal is None:
        ds.close()
        raise ValueError("Missing PRES/TEMP/PSAL in profile file")

    n_prof, n_levels = pres.shape

    # meta arrays (1D length N_PROF)
    cycles = safe_1d(ds, "CYCLE_NUMBER", n_prof)
    juld   = safe_1d(ds, "JULD", n_prof)
    lat    = safe_1d(ds, "LATITUDE", n_prof)
    lon    = safe_1d(ds, "LONGITUDE", n_prof)

    direction = safe_1d_str(ds, "DIRECTION", n_prof)
    data_centre = safe_1d_str(ds, "DATA_CENTRE", n_prof)
    data_state_indicator = safe_1d_str(ds, "DATA_STATE_INDICATOR", n_prof)
    data_mode = safe_1d_str(ds, "DATA_MODE", n_prof)
    positioning_system = safe_1d_str(ds, "POSITIONING_SYSTEM", n_prof)
    config_mission_number = safe_1d(ds, "CONFIG_MISSION_NUMBER", n_prof)

    juld_qc = safe_1d_str(ds, "JULD_QC", n_prof)
    position_qc = safe_1d_str(ds, "POSITION_QC", n_prof)
    profile_pres_qc = safe_1d_str(ds, "PROFILE_PRES_QC", n_prof)
    profile_temp_qc = safe_1d_str(ds, "PROFILE_TEMP_QC", n_prof)
    profile_psal_qc = safe_1d_str(ds, "PROFILE_PSAL_QC", n_prof)

    # measurement arrays [N_PROF, N_LEVELS]
    pres_qc = array_or_none(ds, "PRES_QC")
    temp_qc = array_or_none(ds, "TEMP_QC")
    psal_qc = array_or_none(ds, "PSAL_QC")

    pres_adj = array_or_none(ds, "PRES_ADJUSTED")
    temp_adj = array_or_none(ds, "TEMP_ADJUSTED")
    psal_adj = array_or_none(ds, "PSAL_ADJUSTED")

    pres_err = array_or_none(ds, "PRES_ADJUSTED_ERROR")
    temp_err = array_or_none(ds, "TEMP_ADJUSTED_ERROR")
    psal_err = array_or_none(ds, "PSAL_ADJUSTED_ERROR")

    # scientific calib [N_PROF, 1, N_PARAM(=3)] in strings
    calib = extract_scientific_calib(ds)

    for i in tqdm(range(n_prof), desc="Inserting profiles"):
        pr = Profile(
            platform_number=platform_number,
            cycle_number=int_or_none(cycles[i]),
            juld=float_or_nan(juld[i]),
            timestamp=juld_to_timestamp(ref_dt, juld[i]),
            latitude=float_or_nan(lat[i]),
            longitude=float_or_nan(lon[i]),
            direction=str_or_none(direction[i]),
            data_centre=str_or_none(data_centre[i]),
            data_state_indicator=str_or_none(data_state_indicator[i]),
            data_mode=str_or_none(data_mode[i]),
            positioning_system=str_or_none(positioning_system[i]),
            config_mission_number=int_or_none(config_mission_number[i]) if config_mission_number is not None else None,
            juld_qc=str_or_none(juld_qc[i]),
            position_qc=str_or_none(position_qc[i]),
            profile_pres_qc=str_or_none(profile_pres_qc[i]),
            profile_temp_qc=str_or_none(profile_temp_qc[i]),
            profile_psal_qc=str_or_none(profile_psal_qc[i]),
            scientific_calib=calib.get(i) if calib else None,
        )
        session.add(pr)
        session.flush()  # get pr.id

        # Insert per-depth rows
        for d in range(n_levels):
            row = ProfileMeasurement(
                profile_id=pr.id,
                depth_index=d,
                pres=float_or_nan(pres[i, d]),
                temp=float_or_nan(temp[i, d]),
                psal=float_or_nan(psal[i, d]),
                pres_adjusted=float_or_nan(pres_adj[i, d]) if pres_adj is not None else None,
                temp_adjusted=float_or_nan(temp_adj[i, d]) if temp_adj is not None else None,
                psal_adjusted=float_or_nan(psal_adj[i, d]) if psal_adj is not None else None,
                pres_qc=qc_char_at(pres_qc, i, d),
                temp_qc=qc_char_at(temp_qc, i, d),
                psal_qc=qc_char_at(psal_qc, i, d),
                pres_adjusted_error=float_or_nan(pres_err[i, d]) if pres_err is not None else None,
                temp_adjusted_error=float_or_nan(temp_err[i, d]) if temp_err is not None else None,
                psal_adjusted_error=float_or_nan(psal_err[i, d]) if psal_err is not None else None,
            )
            # Skip entirely empty rows (all NaN)
            if not all_is_nan([row.pres, row.temp, row.psal,
                               row.pres_adjusted, row.temp_adjusted, row.psal_adjusted]):
                session.add(row)

        session.commit()

    ds.close()


def ingest_trajectory(session, platform_number: str, rtraj_nc: Path):
    ds = xr.open_dataset(rtraj_nc, decode_cf=True, decode_times=False)
    ref_dt = parse_ref_datetime(ds)

    # 1D arrays length M (e.g., 1847)
    juld = array_or_none(ds, "JULD")
    lat  = array_or_none(ds, "LATITUDE")
    lon  = array_or_none(ds, "LONGITUDE")
    cyc  = array_or_none(ds, "CYCLE_NUMBER")
    meas_code = array_or_none(ds, "MEASUREMENT_CODE")

    pos_acc = array_or_none(ds, "POSITION_ACCURACY")
    pos_qc  = array_or_none(ds, "POSITION_QC")

    pres = array_or_none(ds, "PRES")
    temp = array_or_none(ds, "TEMP")
    psal = array_or_none(ds, "PSAL")

    pres_adj = array_or_none(ds, "PRES_ADJUSTED")
    temp_adj = array_or_none(ds, "TEMP_ADJUSTED")
    psal_adj = array_or_none(ds, "PSAL_ADJUSTED")

    pres_err = array_or_none(ds, "PRES_ADJUSTED_ERROR")
    temp_err = array_or_none(ds, "TEMP_ADJUSTED_ERROR")
    psal_err = array_or_none(ds, "PSAL_ADJUSTED_ERROR")

    n = len(juld) if juld is not None else 0
    for i in tqdm(range(n), desc="Inserting trajectory points"):
        tp = TrajectoryPoint(
            platform_number=platform_number,
            cycle_number=int_or_none(cyc[i]) if cyc is not None else None,
            juld=float_or_nan(juld[i]),
            timestamp=juld_to_timestamp(ref_dt, juld[i]) if juld is not None else None,
            latitude=float_or_nan(lat[i]) if lat is not None else None,
            longitude=float_or_nan(lon[i]) if lon is not None else None,
            measurement_code=int_or_none(meas_code[i]) if meas_code is not None else None,
            position_accuracy=str_or_none(pos_acc[i]) if pos_acc is not None else None,
            position_qc=str_or_none(pos_qc[i]) if pos_qc is not None else None,
            pres=float_or_nan(pres[i]) if pres is not None else None,
            temp=float_or_nan(temp[i]) if temp is not None else None,
            psal=float_or_nan(psal[i]) if psal is not None else None,
            pres_adjusted=float_or_nan(pres_adj[i]) if pres_adj is not None else None,
            temp_adjusted=float_or_nan(temp_adj[i]) if temp_adj is not None else None,
            psal_adjusted=float_or_nan(psal_adj[i]) if psal_adj is not None else None,
            pres_adjusted_error=float_or_nan(pres_err[i]) if pres_err is not None else None,
            temp_adjusted_error=float_or_nan(temp_err[i]) if temp_err is not None else None,
            psal_adjusted_error=float_or_nan(psal_err[i]) if psal_err is not None else None,
        )
        # Skip if no geo + time + data at all (rare)
        if tp.timestamp or tp.latitude or tp.longitude or tp.pres or tp.temp or tp.psal:
            session.add(tp)
        if i % 1000 == 0:
            session.commit()
    session.commit()

    # Events: arrays length N_PROF (99) for many *_STATUS pairs
    EVENT_KEYS = [
        "ASCENT_START", "ASCENT_END", "DESCENT_START", "DESCENT_END",
        "TRANSMISSION_START", "TRANSMISSION_END",
        "FIRST_STABILIZATION", "PARK_START", "PARK_END",
        "DEEP_PARK_START", "DEEP_DESCENT_END", "DEEP_ASCENT_START",
        "FIRST_MESSAGE", "FIRST_LOCATION", "LAST_MESSAGE", "LAST_LOCATION"
    ]
    for base in EVENT_KEYS:
        juld_key = f"JULD_{base}"
        status_key = f"JULD_{base}_STATUS"
        jarr = array_or_none(ds, juld_key)
        sarr = array_or_none(ds, status_key)
        carr = array_or_none(ds, "CYCLE_NUMBER")  # reuse cycle_number len = N_PROF
        if jarr is None or carr is None:
            continue
        n_ev = len(jarr)
        for i in range(n_ev):
            jv = jarr[i]
            ts = juld_to_timestamp(ref_dt, jv)
            ev = TrajectoryEvent(
                platform_number=platform_number,
                cycle_number=int_or_none(carr[i]),
                event_name=base,
                juld=float_or_nan(jv),
                timestamp=ts,
                status=str_or_none(sarr[i]) if sarr is not None else None,
            )
            if ev.juld is not None and not np.isnan(ev.juld):
                session.add(ev)
        session.commit()

    ds.close()


def ingest_tech(session, platform_number: str, tech_nc: Path):
    ds = xr.open_dataset(tech_nc, decode_cf=True, decode_times=False)
    names = array_or_none(ds, "TECHNICAL_PARAMETER_NAME")
    values = array_or_none(ds, "TECHNICAL_PARAMETER_VALUE")
    cycles = array_or_none(ds, "CYCLE_NUMBER")
    n = len(names) if names is not None else 0

    for i in tqdm(range(n), desc="Inserting tech parameters"):
        tp = TechnicalParameter(
            platform_number=platform_number,
            cycle_number=int_or_none(cycles[i]) if cycles is not None else None,
            parameter_name=str_or_none(names[i]),
            parameter_value=str_or_none(values[i]),
        )
        if tp.parameter_name:
            session.add(tp)
        if i % 2000 == 0:
            session.commit()
    session.commit()
    ds.close()


# ---------- Utility pieces used above ----------
def ensure_number(x):
    try:
        if x is None:
            return None
        if isinstance(x, (list, np.ndarray)) and len(x) == 1:
            x = x[0]
        return float(x)
    except Exception:
        try:
            return int(x)
        except Exception:
            return None

def ensure_float(x):
    try:
        return float(np.array(x).item())
    except Exception:
        try:
            return float(x)
        except Exception:
            return None

def juld_to_ts_from_calendar_str(cal_str: Optional[str]) -> Optional[datetime]:
    """
    Many meta fields are ISO-like strings; just parse gracefully.
    Return None if empty/invalid.
    """
    if not cal_str:
        return None
    try:
        return dtparser.parse(cal_str)
    except Exception:
        return None

def max_len_arrays(ds: xr.Dataset, keys: List[str]) -> int:
    m = 0
    for k in keys:
        arr = array_or_none(ds, k)
        if arr is not None:
            m = max(m, len(arr))
    return m

def pick_idx(ds: xr.Dataset, key: str, i: int) -> Optional[str]:
    arr = array_or_none(ds, key)
    if arr is None:
        return None
    try:
        return to_str_or_none(arr[i])
    except Exception:
        return None

def int_or_none(x):
    try:
        if x is None:
            return None
        return int(np.array(x).item())
    except Exception:
        try:
            return int(x)
        except Exception:
            return None

def float_or_nan(x):
    try:
        v = float(np.array(x).item())
        return v
    except Exception:
        try:
            return float(x)
        except Exception:
            return np.nan

def str_or_none(x):
    try:
        if x is None:
            return None
        if isinstance(x, (bytes, bytearray)):
            return x.decode("utf-8", "ignore")
        if isinstance(x, np.ndarray):
            return to_str_or_none(x)
        s = str(x)
        return s if s.strip() != "" else None
    except Exception:
        return None

def qc_char_at(qc_arr: Optional[np.ndarray], i: int, j: int) -> Optional[str]:
    if qc_arr is None:
        return None
    try:
        val = qc_arr[i, j]
        return str_or_none(val)
    except Exception:
        return None

def safe_1d(ds: xr.Dataset, key: str, n: int) -> np.ndarray:
    arr = array_or_none(ds, key)
    if arr is None:
        return np.array([None] * n)
    return arr

def safe_1d_str(ds: xr.Dataset, key: str, n: int) -> np.ndarray:
    arr = array_or_none(ds, key)
    if arr is None:
        return np.array([None] * n, dtype=object)
    # Normalize to string-ish
    out = np.empty(len(arr), dtype=object)
    for i in range(len(arr)):
        out[i] = str_or_none(arr[i])
    return out

def build_config_params(names: Optional[np.ndarray], vals: Optional[np.ndarray]):
    if names is None or vals is None:
        return None
    names = np.array(names).tolist()
    vals = np.array(vals)
    # vals could be [1, N] or [N]
    if vals.ndim == 2 and vals.shape[0] == 1:
        vals = vals[0]
    vals = vals.tolist()
    d = {}
    for i, k in enumerate(names):
        kk = to_str_or_none(k)
        if kk is None:
            continue
        # keep as float if numeric, else string
        try:
            d[kk] = float(vals[i])
        except Exception:
            d[kk] = vals[i]
    return d

def extract_scientific_calib(ds: xr.Dataset) -> Dict[int, List[Dict[str, Any]]]:
    """
    Return {profile_index: [ {parameter, equation, coefficient, comment, date}, ...]}
    Based on arrays shaped [N_PROF, 1, N_PARAM] (strings).
    """
    out = {}
    params = array_or_none(ds, "PARAMETER")
    eqn    = array_or_none(ds, "SCIENTIFIC_CALIB_EQUATION")
    coeff  = array_or_none(ds, "SCIENTIFIC_CALIB_COEFFICIENT")
    comm   = array_or_none(ds, "SCIENTIFIC_CALIB_COMMENT")
    date   = array_or_none(ds, "SCIENTIFIC_CALIB_DATE")

    if params is None:
        return out

    # params has shape [N_PROF, 1, N_PARAM] — squeeze
    params = np.squeeze(params, axis=1) if params.ndim == 3 and params.shape[1] == 1 else params
    if eqn is not None and eqn.ndim == 3 and eqn.shape[1] == 1:
        eqn = np.squeeze(eqn, axis=1)
    if coeff is not None and coeff.ndim == 3 and coeff.shape[1] == 1:
        coeff = np.squeeze(coeff, axis=1)
    if comm is not None and comm.ndim == 3 and comm.shape[1] == 1:
        comm = np.squeeze(comm, axis=1)
    if date is not None and date.ndim == 3 and date.shape[1] == 1:
        date = np.squeeze(date, axis=1)

    n_prof, n_param = params.shape
    for i in range(n_prof):
        items = []
        for j in range(n_param):
            items.append({
                "parameter": str_or_none(params[i, j]),
                "equation": str_or_none(eqn[i, j]) if eqn is not None else None,
                "coefficient": str_or_none(coeff[i, j]) if coeff is not None else None,
                "comment": str_or_none(comm[i, j]) if comm is not None else None,
                "date": str_or_none(date[i, j]) if date is not None else None,
            })
        out[i] = items
    return out

# ---------- Main ----------
def main():
    print(f"Connecting to {PGURL}")
    Base.metadata.create_all(engine)
    session = SessionLocal()

    # 1) Meta (returns platform_number)
    platform_number = ingest_meta(session, META_NC)
    print("Meta ingested:", platform_number)

    # 2) Profiles (profile rows + per-depth measurements)
    ingest_profiles(session, platform_number, PROF_NC)
    print("Profiles ingested.")

    # 3) Trajectory (dense time series + cycle events)
    ingest_trajectory(session, platform_number, RTRAJ_NC)
    print("Trajectory ingested.")

    # 4) Technical parameters (per-cycle diagnostics)
    ingest_tech(session, platform_number, TECH_NC)
    print("Technical parameters ingested.")

    session.close()
    print("Done.")

if __name__ == "__main__":
    main()

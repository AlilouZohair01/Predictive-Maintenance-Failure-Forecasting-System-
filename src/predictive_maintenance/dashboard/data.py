"""Read-only FD001 observations; no future observations or RUL truth in inference inputs."""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from ..api.schemas import SENSORS

SENSOR_INFO = {
    "s_2": ("LPC outlet temperature", "°R"), "s_3": ("HPC outlet temperature", "°R"),
    "s_4": ("LPT outlet temperature", "°R"), "s_7": ("HPC outlet pressure", "psia"),
    "s_8": ("Fan speed", "rpm"), "s_11": ("HPC static pressure", "psia"),
    "s_12": ("Fuel-flow / pressure ratio", "pps/psi"), "s_13": ("Corrected fan speed", "rpm"),
    "s_14": ("Corrected core speed", "rpm"), "s_15": ("Bypass ratio", "ratio"),
    "s_17": ("Bleed enthalpy indicator", "native"), "s_20": ("HPT coolant bleed", "lbm/s"),
    "s_21": ("LPT coolant bleed", "lbm/s"),
}


class Dataset:
    def __init__(self, path):
        path = Path(path)
        self.fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        columns = ["unit", "cycle", "setting_1", "setting_2", "setting_3"] + [f"s_{i}" for i in range(1, 22)]
        data = pd.read_csv(path, sep=r"\s+", header=None)
        if data.shape[1] != 26:
            raise ValueError("Expected the original 26-column FD001 test observations")
        data.columns = columns
        if not np.isfinite(data[["unit", "cycle"] + SENSORS].to_numpy()).all():
            raise ValueError("Dataset contains non-finite observations")
        self.engines = {}
        for unit, frame in data.groupby("unit", sort=True):
            cycles = frame.cycle.to_numpy()
            if int(unit) != unit or unit < 1 or not np.array_equal(cycles, np.arange(1, len(frame) + 1)):
                raise ValueError("Expected ordered consecutive engine histories starting at cycle 1")
            self.engines[int(unit)] = frame.reset_index(drop=True)
        if not self.engines:
            raise ValueError("Dataset is empty")

    def length(self, unit):
        if isinstance(unit, bool) or not isinstance(unit, (int, np.integer)) or unit not in self.engines:
            raise ValueError("Select an available engine")
        return len(self.engines[unit])

    def prefix(self, unit, cycle):
        length = self.length(unit)
        if isinstance(cycle, bool) or not isinstance(cycle, (int, np.integer)) or not 1 <= cycle <= length:
            raise ValueError("Select an observed cycle for this engine")
        return self.engines[unit].iloc[:cycle].copy()

    def readings(self, unit, cycle):
        rows = self.prefix(unit, cycle)[["cycle"] + SENSORS].to_dict("records")
        for row in rows:
            row["cycle"] = int(row["cycle"])
        return rows

"""Causal, per-zone streaming anomaly detector.

The evaluator feeds one reading at a time; only current/past observations
are used. Scores are always finite and between 0 and 1.
"""
from collections import defaultdict, deque
import numpy as np


class Detector:
    FIELDS = (
        "temperature_c", "humidity_pct", "vibration_g",
        "acoustic_db", "air_pressure_kpa", "co2_ppm"
    )
    SCALE = np.array([0.23, 0.55, 0.010, 0.32, 0.05, 11.0])

    def __init__(self):
        self.history = defaultdict(lambda: deque(maxlen=241))
        self.counts = defaultdict(int)

    def update(self, reading: dict) -> float:
        zone = str(reading.get("bunker_zone", "default"))
        values = np.array([float(reading[k]) for k in self.FIELDS])
        history = self.history[zone]
        previous = np.asarray(history, dtype=float) if history else np.empty((0, 6))
        i = len(history)
        count = self.counts[zone]
        self.counts[zone] += 1

        if count > 30:
            baseline = np.median(previous[max(0, i - 240):max(1, i - 15)], axis=0)
        else:
            baseline = np.median(previous, axis=0) if i else values

        z = (values - baseline) / self.SCALE
        point = np.max(np.abs(z)) > 5.0
        collective = (min(abs(z[2]), abs(z[3])) > 2.5
                      and max(abs(z[2]), abs(z[3]), abs(z[5])) > 4.0)

        drift = False
        for window, threshold in ((12, 4.0), (45, 4.0)):
            if count < window:
                continue
            recent = np.vstack((previous[-(window - 1):], values))
            recent_mean = np.mean(recent, axis=0)
            if count > window + 15:
                old_end = max(1, i - window - 15)
                old_start = max(0, i - 240)
                old = previous[old_start:old_end]
                reference = np.median(old, axis=0) if len(old) else baseline
            else:
                reference = baseline
            if np.max(np.abs((recent_mean - reference) / self.SCALE)) > threshold:
                drift = True
                break

        history.append(values)
        return 1.0 if point or collective or drift else 0.0

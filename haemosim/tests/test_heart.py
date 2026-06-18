import numpy as np

from core.heart import HeartModel


def _settled_results(duration=10.0, dt=0.001, settle_time=5.0):
    model = HeartModel()
    results = model.simulate(duration=duration, dt=dt)
    mask = results["time"] >= settle_time
    return model, results, mask


def test_default_aortic_pressure_stays_physiological_after_settling():
    _, results, settled = _settled_results()

    pressure = results["P_aortic"][settled]

    assert np.min(pressure) >= 60.0
    assert np.max(pressure) <= 140.0


def test_default_stroke_volume_is_physiological_after_settling():
    model, results, _ = _settled_results()
    period = 60.0 / model.heart_rate
    last_cycle = results["time"] >= (results["time"][-1] - period)

    stroke_volume = np.max(results["V_lv"][last_cycle]) - np.min(results["V_lv"][last_cycle])

    assert 60.0 <= stroke_volume <= 100.0

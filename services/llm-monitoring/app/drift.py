"""Small-sample statistical drift screening using a reference and current numeric window."""
import math
from statistics import mean, pstdev

def compare_windows(reference: list[float], current: list[float], threshold: float = 2.0) -> dict:
    if len(reference) < 5 or len(current) < 5:
        raise ValueError('At least 5 numeric observations are required in each window.')
    if any(not math.isfinite(float(x)) for x in reference + current):
        raise ValueError('All observations must be finite numbers.')
    ref_mean, cur_mean = mean(reference), mean(current)
    ref_std = pstdev(reference)
    scale = ref_std if ref_std > 1e-9 else max(abs(ref_mean) * 0.01, 1e-6)
    z_score = (cur_mean - ref_mean) / scale
    return {'reference_count':len(reference),'current_count':len(current),'reference_mean':round(ref_mean,6),
            'current_mean':round(cur_mean,6),'reference_stddev':round(ref_std,6),
            'mean_shift_z_score':round(z_score,4),'threshold':threshold,
            'drift_detected':abs(z_score) >= threshold,
            'method':'absolute mean shift divided by reference population standard deviation',
            'caveat':'Screening signal, not a formal hypothesis test; validate with domain-specific baselines.'}

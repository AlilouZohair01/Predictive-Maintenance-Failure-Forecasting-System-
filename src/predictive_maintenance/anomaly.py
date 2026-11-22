"""Causal FD001 anomaly scoring. Failure labels never enter these functions."""
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

ALGORITHMS = ('isolation_forest', 'autoencoder', 'zscore')


def validate_frame(frame, sensors):
    frame = frame.sort_values(['unit_number', 'time_in_cycles']).reset_index(drop=True).copy()
    if frame.duplicated(['unit_number', 'time_in_cycles']).any():
        raise ValueError('Duplicate engine/cycle observations')
    if not np.isfinite(frame[sensors].to_numpy()).all():
        raise ValueError('Sensor values must be finite')
    for _, group in frame.groupby('unit_number'):
        if not np.all(np.diff(group.time_in_cycles) == 1):
            raise ValueError('Cycles must be consecutive within an engine')
    return frame


def windows(frame, scaled, length=30):
    values, endpoints = [], []
    for _, group in frame.groupby('unit_number', sort=False):
        idx = group.index.to_numpy()
        for end in range(length - 1, len(idx)):
            values.append(scaled[idx[end-length+1:end+1]].reshape(-1))
            endpoints.append(idx[end])
    width = length * scaled.shape[1]
    return np.asarray(values, dtype=np.float32).reshape(-1, width), np.asarray(endpoints, dtype=int)


def causal_mean(values, units, width):
    return pd.Series(values).groupby(np.asarray(units), sort=False).transform(
        lambda s: s.rolling(width, min_periods=1).mean()).to_numpy()


def sustained_flags(flags, units, count):
    """Alert becomes true on the Nth consecutive flag; never backdate it."""
    return pd.Series(np.asarray(flags, dtype=int)).groupby(np.asarray(units), sort=False).transform(
        lambda s: s.rolling(count, min_periods=count).sum().eq(count)).to_numpy(dtype=bool)


def score_cycles(frame, sensors, scaler, forest, autoencoder, length=30, smoothing=5):
    frame = validate_frame(frame, sensors)
    scaled = scaler.transform(frame[sensors]).astype(np.float32)
    matrix, endpoints = windows(frame, scaled, length)
    result = frame[['unit_number', 'time_in_cycles']].copy()
    result['isolation_forest_raw'] = -forest.score_samples(scaled)
    result['zscore_raw'] = np.mean(scaled ** 2, axis=1)
    result['autoencoder_raw'] = np.nan
    if len(matrix):
        reconstruction = autoencoder.predict(matrix, batch_size=256, verbose=0)
        result.loc[endpoints, 'autoencoder_raw'] = np.mean((matrix - reconstruction) ** 2, axis=1)
    for name in ALGORITHMS:
        # Same evaluation start for every detector. First 29 cycles are warm-up.
        valid = result.time_in_cycles >= length
        raw = result[f'{name}_raw'].where(valid)
        result[f'{name}_score'] = causal_mean(raw, result.unit_number, smoothing)
    return result


def apply_threshold(scores, name, threshold, consecutive=3):
    flag = (scores[f'{name}_score'] > threshold).to_numpy()
    alert = sustained_flags(flag, scores.unit_number, consecutive)
    return flag, alert


def evaluate(scores, rul, name, threshold, failure_window=30, consecutive=3):
    flag, alert = apply_threshold(scores, name, threshold, consecutive)
    valid = scores[f'{name}_score'].notna().to_numpy()
    rul = np.asarray(rul)
    y = rul <= failure_window
    healthy = valid & (scores.time_in_cycles.to_numpy() <= 50)
    distant = valid & (rul > 125)
    rate = lambda pred, mask: float(np.mean(pred[mask])) if mask.any() else np.nan
    return {
        'n_cycles': int(valid.sum()), 'positive_cycles': int(y[valid].sum()),
        'positive_prevalence': float(y[valid].mean()),
        'average_precision': float(average_precision_score(y[valid], scores.loc[valid, f'{name}_score'])) if y[valid].any() else np.nan,
        'cycle_precision': float(precision_score(y[valid], flag[valid], zero_division=0)),
        'cycle_recall': float(recall_score(y[valid], flag[valid], zero_division=0)),
        'cycle_f1': float(f1_score(y[valid], flag[valid], zero_division=0)),
        'alert_precision': float(precision_score(y[valid], alert[valid], zero_division=0)),
        'alert_recall': float(recall_score(y[valid], alert[valid], zero_division=0)),
        'alert_f1': float(f1_score(y[valid], alert[valid], zero_division=0)),
        'healthy_cycle_fpr': rate(flag, healthy), 'healthy_alert_fpr': rate(alert, healthy),
        'rul_gt125_cycle_fpr': rate(flag, distant), 'rul_gt125_alert_fpr': rate(alert, distant),
    }


def select_thresholds(validation_scores, validation_rul, name, consecutive=3):
    series = validation_scores[f'{name}_score']
    healthy = series[(validation_scores.time_in_cycles <= 50) & series.notna()]
    if not len(healthy):
        raise ValueError('No healthy validation scores')
    primary = float(healthy.quantile(0.98))
    # Fixed search budget, validation only. Optimize sustained-alert F1 for RUL <= 30.
    candidates = np.unique(np.r_[np.nextafter(series.min(), -np.inf),
                                 series.dropna().quantile(np.linspace(0, 1, 201)).to_numpy()])
    trials = []
    for threshold in candidates:
        metrics = evaluate(validation_scores, validation_rul, name, threshold, 30, consecutive)
        trials.append({'threshold': float(threshold), 'alert_f1': metrics['alert_f1'],
                       'healthy_alert_fpr': metrics['healthy_alert_fpr']})
    trials = pd.DataFrame(trials).sort_values(['alert_f1', 'healthy_alert_fpr', 'threshold'], ascending=[False, True, False])
    return {'healthy_q98': primary, 'validation_f1': float(trials.iloc[0].threshold)}, trials


def lead_times(scores, rul, name, threshold, consecutive=3):
    _, alert = apply_threshold(scores, name, threshold, consecutive)
    table = scores[['unit_number', 'time_in_cycles']].copy()
    table['rul'] = np.asarray(rul)
    table['alert'] = alert
    rows = []
    for unit, group in table.groupby('unit_number'):
        hits = group[group.alert]
        first = hits.iloc[0] if len(hits) else None
        rows.append({'unit_number': unit, 'observed_alert': first is not None,
                     'first_alert_cycle': float(first.time_in_cycles) if first is not None else np.nan,
                     'lead_time_cycles': float(first.rul) if first is not None else np.nan,
                     'observed_failure_window': bool(group.rul.le(30).any())})
    return pd.DataFrame(rows)


def predict_saved(frame, model_dir):
    """Score complete consecutive histories per engine; includes warm-up rows.

    Supply the full history when calling repeatedly so smoothing and alert state
    are preserved. Scores are anomaly measures, not failure probabilities.
    """
    from pathlib import Path
    import json
    import joblib
    import tensorflow as tf
    model_dir = Path(model_dir)
    config = json.loads((model_dir / 'metadata.json').read_text())
    bundle = joblib.load(model_dir / 'detectors.pkl')
    ae = tf.keras.models.load_model(model_dir / 'autoencoder.keras', compile=False)
    scored = score_cycles(frame, config['sensors'], bundle['scaler'], bundle['forest'], ae,
                          config['window_length'], config['smoothing'])
    for name in ALGORITHMS:
        flag, alert = apply_threshold(scored, name, config['thresholds'][name]['healthy_q98'], config['consecutive'])
        scored[f'{name}_flag'] = flag
        scored[f'{name}_alert'] = alert
    return scored

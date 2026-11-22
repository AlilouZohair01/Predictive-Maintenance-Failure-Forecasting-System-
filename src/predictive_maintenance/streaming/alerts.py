"""Deterministic policy transitions; separate from learned model outputs."""


def transition(rule, previous, cycle, rul, anomaly, policy):
    state = dict(previous or {"active": False, "clear_streak": 0, "last_raised_cycle": None, "alert_id": None})
    if rule == "low_rul":
        trigger = rul is not None and rul < policy["rul_trigger_below"]
        clear = rul is not None and rul >= policy["rul_clear_at"]
    else:
        trigger = anomaly is not None and anomaly.get("sustained_alert") is True
        normal = anomaly is not None and anomaly.get("is_anomaly") is False
        state["clear_streak"] = state["clear_streak"] + 1 if normal else 0
        clear = state["clear_streak"] >= policy["anomaly_clear_cycles"]
    event = None
    if state["active"] and clear:
        state["active"] = False
        event = "resolved"
    elif not state["active"] and trigger:
        last = state["last_raised_cycle"]
        if last is None or cycle - last >= policy["cooldown_cycles"]:
            state.update(active=True, clear_streak=0, last_raised_cycle=cycle)
            event = "raised"
    return state, event

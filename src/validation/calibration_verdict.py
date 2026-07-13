"""Prescriptive calibration verdict: *what to do* about a posterior.

TARP (``tarp_test``) reports whether credible intervals are the right *width* —
the calibration axis. It cannot, on its own, tell an under-fit flow (the true
posterior is tighter, so *train*) from an information-limited one (the wide
posterior is honest, so *accept*): both read as "under-confident". The coverage
probe (``online_probe``) supplies the missing *accuracy* axis via RMSE. This
module fuses the two into a single action, keeping TARP and the probe otherwise
decoupled.

Actions
-------
    accept          calibrated and informative — nothing to do
    accept-limited  calibrated but information-limited — needs better data, not training
    recalibrate     uniform scale error — cheap post-hoc temperature (or train)
    train           shape error (mixed) or under-fit — a scalar rescale can't fix it

Why "mixed -> train": a temperature is a single scalar; it can only correct a
uniform over/under-confidence. An ECP curve that crosses the diagonal is a
*shape* error, so no scalar rescale makes it calibrated — only retraining can.
"""

import numpy as np

# RMSE is reported in prior-width units. A parameter with no information has a
# posterior std equal to that of the uniform prior = 1/sqrt(12) ~ 0.289. We call
# a parameter "informative" only when its RMSE sits well below that no-info floor.
UNIFORM_STD = 0.2887
INFORMATIVE_RMSE = 0.15  # < ~half the no-information std


def prescribe(tarp: dict, probe: dict | None = None,
              informative_rmse: float = INFORMATIVE_RMSE) -> dict:
    """Return ``{"action", "reason"}`` from a TARP summary (+ optional probe).

    ``tarp`` is the summary dict from ``tarp_test.run_tarp``; ``probe`` is the
    summary from ``online_probe.run_probe`` (its ``rmse`` list, in prior-width
    units). Without the probe, only the calibration axis is available.
    """
    verdict = tarp.get("verdict")

    if verdict == "mixed":
        return {
            "action": "train",
            "reason": "ECP curve crosses the diagonal (shape error); a scalar "
                      "temperature cannot fix it — retrain.",
        }

    mean_rmse = None
    if probe is not None and probe.get("rmse") is not None:
        mean_rmse = float(np.mean(np.asarray(probe["rmse"])))

    if verdict == "calibrated":
        if mean_rmse is None:
            return {"action": "accept", "reason": "calibrated (TARP); RMSE unknown."}
        if mean_rmse <= informative_rmse:
            return {"action": "accept",
                    "reason": f"calibrated and informative (mean RMSE {mean_rmse:.3f})."}
        return {"action": "accept-limited",
                "reason": f"calibrated but information-limited (mean RMSE {mean_rmse:.3f} "
                          f"toward no-info floor {UNIFORM_STD:.3f}); improve data, not training."}

    # Significant, uniform-direction miscalibration (over- or under-confident).
    direction = "widen (T>1)" if verdict == "over-confident" else "sharpen (T<1)"
    action = {
        "action": "recalibrate",
        "reason": f"{verdict}; uniform scale error — {direction} via temperature "
                  f"(cheap) or train.",
    }
    # Under-confident *with* poor accuracy is more likely under-fitting than a
    # pure scale error: temperature would manufacture false precision, so train.
    if verdict == "under-confident" and mean_rmse is not None and mean_rmse > informative_rmse:
        action = {
            "action": "train",
            "reason": f"under-confident with high RMSE ({mean_rmse:.3f}) — likely "
                      f"under-fit; train to sharpen (cuts RMSE and coverage error together).",
        }
    return action

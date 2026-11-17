"""Prepare a fitted model for timed, batch-by-batch inference.

The inference dashboards time every batch, so the numbers on screen must measure the prediction
itself, not the cost of handing work to threads.
"""


def prepare_for_serving(model):
    """Switch a trained model to single-threaded predict (it is updated in place).

    Training uses every core, but for small batches joblib's thread dispatch costs several times
    more than the prediction itself, which would make latency/throughput numbers meaningless.

    Args:
        model: A fitted scikit-learn estimator.

    Returns:
        The same model object, now with ``n_jobs=1`` if it has that parameter.
    """
    if "n_jobs" in model.get_params():
        model.set_params(n_jobs=1)
    return model

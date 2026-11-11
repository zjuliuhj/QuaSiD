"""Normalization policy shared by time-series dataset loaders."""

from sklearn.preprocessing import MinMaxScaler, StandardScaler


_ZSCORE = "zscore"
_MINMAX = "minmax"
_NONE = "none"
_VALID = {_ZSCORE, _MINMAX, _NONE}


def resolve_normalization(normalization):
    """Return the selected config method: `zscore`, `minmax`, or `none`."""
    if normalization is None:
        return _NONE

    method = str(normalization).strip().lower()
    if method not in _VALID:
        supported = ", ".join(sorted(_VALID))
        raise ValueError(f"normalization must be one of: {supported}. Got {normalization!r}.")
    return method


def make_normalizer(method):
    """Return an unfitted scaler; dataset loaders fit it only on their train split."""
    if method == _NONE:
        return None
    if method == _ZSCORE:
        return StandardScaler()
    if method == _MINMAX:
        return MinMaxScaler(feature_range=(-1.0, 1.0))
    raise ValueError(f"Unsupported normalization method: {method!r}.")

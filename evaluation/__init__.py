def build_evaluator(*args, **kwargs):
    """Lazily construct the task evaluator without importing metric stacks at package load."""
    from evaluation.runners import build_evaluator as _build_evaluator

    return _build_evaluator(*args, **kwargs)


__all__ = ["build_evaluator"]

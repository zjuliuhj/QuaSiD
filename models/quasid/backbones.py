import inspect

from models.quasid.dit_solver_v5 import DiTSolverV5


BACKBONE_REGISTRY = {
    "dit_solver_v5": DiTSolverV5,
}


def build_backbone(name, **kwargs):
    """Instantiate a registered DiT backbone and reject unsupported config keys."""
    if name not in BACKBONE_REGISTRY:
        available = ", ".join(sorted(BACKBONE_REGISTRY))
        raise ValueError(f"Unknown backbone `{name}`. Available backbones: {available}")
    cls = BACKBONE_REGISTRY[name]
    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
    unknown = sorted(set(kwargs) - accepted)
    if unknown:
        raise ValueError(f"Unsupported kwargs for backbone `{name}`: {unknown}")
    return cls(**kwargs)

from .dbm import DynamicBicycleModel, DynamicParams, CarState, CarAction
from .models import AdaptDataset, ParamAdaptModel

def __getattr__(name):
    if name == "DynamicsJax":
        from .nn_dynamics import DynamicsJax
        return DynamicsJax
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

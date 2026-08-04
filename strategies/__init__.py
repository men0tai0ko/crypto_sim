from .base import Strategy, Context
from .baselines import BuyHoldBTC, EqualWeight, DCA
from .trend import DonchianTrend, SMACross, FilteredEqualWeight
from .rotation import MomentumRotation, RSIMeanReversion
from .regime_switch import RegimeSwitching

__all__ = [
    "Strategy", "Context",
    "BuyHoldBTC", "EqualWeight", "DCA",
    "DonchianTrend", "SMACross", "FilteredEqualWeight",
    "MomentumRotation", "RSIMeanReversion",
    "RegimeSwitching",
]

"""
核心集选择方法包
"""
from .coreset_base import CoresetSelector, ContinualLearningFramework
from .methods.random import RandomSelector
from .methods.greedy import GreedySelector
from .methods.csrel import CSReLSelector
from .methods.bcsr import BCSRSelector
from .methods.ensemble import EnsembleSelector

# 方法注册表
METHOD_REGISTRY = {
    'random': RandomSelector,
    'greedy': GreedySelector,
    'csrel': CSReLSelector,
    'bcsr': BCSRSelector,
    'ensemble': EnsembleSelector,
}


def get_selector(method_name: str, **kwargs):
    """根据名称获取选择器实例"""
    if method_name not in METHOD_REGISTRY:
        raise ValueError(
            f"Unknown method: {method_name}. "
            f"Available: {list(METHOD_REGISTRY.keys())}"
        )
    return METHOD_REGISTRY[method_name](**kwargs)

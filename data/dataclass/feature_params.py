"""
Class providing feature params for training pipeline
"""

from dataclasses import dataclass, field


@dataclass()
class FeatureParams:
    target_name: str = field()

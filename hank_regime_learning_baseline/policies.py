from __future__ import annotations

import numpy as np

from hank_learning_policy_baseline.policies import BasePolicy


class MisspecifiedClassicalRulePolicy(BasePolicy):
    def rate(self, observation: np.ndarray, info: dict) -> float:
        lower, upper = info["rate_bounds"]
        return float(np.clip(info["misspecified_filtered_rule_rate"], lower, upper))

from __future__ import annotations

import pandas as pd


def solve_sequence_space_jacobian(bundle, horizon):
    outputs = ["Y", "C", "pi", "i", "N", "w", "A", "B", "output_gap"]
    jacobian = bundle["model"].solve_jacobian(
        bundle["ss"],
        bundle["unknowns"],
        bundle["targets"],
        inputs=["monetary_policy_shock"],
        outputs=outputs,
        T=horizon,
    )

    rows = []
    for output in jacobian.outputs:
        matrix = jacobian.nesteddict[output]["monetary_policy_shock"]
        for response_period in range(matrix.shape[0]):
            for shock_period in range(matrix.shape[1]):
                rows.append({
                    "output": output,
                    "input": "monetary_policy_shock",
                    "response_period": response_period,
                    "shock_period": shock_period,
                    "value": matrix[response_period, shock_period],
                })
    return pd.DataFrame(rows)

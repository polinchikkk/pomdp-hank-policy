from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from regime_switching_baseline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Запуск этапа 5: HANK со скрытыми режимами при неполной наблюдаемости."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/hank_regime_switching_stage5",
        help="Directory for generated outputs.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help="Необязательный список сценариев для запуска.",
    )
    parser.add_argument(
        "--article-information-regimes",
        action="store_true",
        help=(
            "Использовать чистую матрицу для основного текста: "
            "базовые макроэкономические наблюдения и макронаблюдения "
            "с шумными распределительными сигналами."
        ),
    )
    args = parser.parse_args()
    run_pipeline(
        output_dir=args.output_dir,
        scenario_names=args.scenarios,
        use_article_information_scenarios=bool(args.article_information_regimes),
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _ensure_src_on_path() -> None:
    import sys

    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> int:
    _ensure_src_on_path()

    import pandas as pd

    from paddy_yield.config import PipelineConfig
    from paddy_yield.data import load_dataset, prepare_dataframe

    parser = argparse.ArgumentParser(description="Profile and sanity-check the paddy dataset.")
    parser.add_argument("--data-path", required=True, help="Path to paddy_with_yield.csv")
    parser.add_argument("--out", default="outputs/data_profile.json", help="Where to write JSON output.")
    args = parser.parse_args()

    cfg = PipelineConfig()
    df_raw = load_dataset(args.data_path)
    prepared = prepare_dataframe(df_raw, cfg)
    df = prepared.df

    profile = {
        "raw_shape": [int(df_raw.shape[0]), int(df_raw.shape[1])],
        "prepared_shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": list(df.columns),
        "feature_cols": prepared.feature_cols,
        "target_col": cfg.target_col,
        "missing_by_col_raw": {k: int(v) for k, v in df_raw.isna().sum().to_dict().items()},
        "prepared_numeric_min_max": df.describe(numeric_only=True).loc[["min", "max"]].to_dict(),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(profile, indent=2))
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


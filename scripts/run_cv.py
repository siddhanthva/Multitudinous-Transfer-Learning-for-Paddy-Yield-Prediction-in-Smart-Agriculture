from __future__ import annotations

import argparse
from pathlib import Path


def _ensure_src_on_path() -> None:
    import sys

    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> int:
    _ensure_src_on_path()

    from paddy_yield.config import PipelineConfig
    from paddy_yield.data import load_dataset, prepare_dataframe
    from paddy_yield.train import run_cv_bilstm, run_cv_mtl

    parser = argparse.ArgumentParser(description="5-fold CV for paddy yield prediction.")
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to paddy_with_yield.csv",
    )
    parser.add_argument(
        "--model",
        default="both",
        choices=["bilstm", "mtl", "both"],
        help="Which model(s) to run.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs",
        help="Output directory (default: outputs).",
    )
    args = parser.parse_args()

    cfg = PipelineConfig()
    df_raw = load_dataset(args.data_path)
    prepared = prepare_dataframe(df_raw, cfg)
    df = prepared.df
    feature_cols = prepared.feature_cols

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.model in ("bilstm", "both"):
        cv_bi = run_cv_bilstm(df, feature_cols, cfg, out_dir=out_root / "bilstm")
        print("Bi-LSTM summary:")
        print(cv_bi.summary())

    if args.model in ("mtl", "both"):
        cv_mtl = run_cv_mtl(df, feature_cols, cfg, out_dir=out_root / "mtl")
        print("MTL summary:")
        print(cv_mtl.summary())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


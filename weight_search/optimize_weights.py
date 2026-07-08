import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from exchange.client import get_client
from scoring_bot import (
    END,
    EXPOSURE_LEVELS,
    FEE_RATE,
    INITIAL_CASH,
    INTERVAL,
    PROJECT_ROOT,
    START,
    SYMBOL,
    TESTNET,
    ZONE_EDGES,
    build_feature_frame,
    load_mvrv_series,
)


WEIGHT_KEYS = ("sma_dist", "ema_slope", "rsi", "exnet", "mvrv")
RESULTS_DIR = PROJECT_ROOT / "weight_search"
BEST_WEIGHTS_PATH = RESULTS_DIR / "best_weights.json"


def normalize_weights(raw_weights):
    total = sum(raw_weights.values())
    if total <= 0:
        return {key: 1.0 / len(raw_weights) for key in raw_weights}
    return {key: value / total for key, value in raw_weights.items()}


def apply_weights(feat, weights):
    working = feat.copy()
    working["combined_signal"] = sum(
        weights[key] * working[f"{key}_scaled"] for key in WEIGHT_KEYS
    )
    working["bo_exposure"] = pd.cut(
        working["combined_signal"],
        bins=[-np.inf] + ZONE_EDGES + [np.inf],
        labels=EXPOSURE_LEVELS,
        include_lowest=True,
    ).astype(float)
    return working


def backtest_window(feat, weights, mask):
    working = apply_weights(feat, weights)
    position = working["bo_exposure"].astype(float).fillna(0.0)
    position_change = position.diff().abs().fillna(0.0)
    transaction_cost = position_change * FEE_RATE
    strategy_return = position.shift(1).fillna(0.0) * working["close"].pct_change().fillna(0.0) - transaction_cost

    window_returns = strategy_return.loc[mask]
    equity_curve = (1 + window_returns).cumprod() * INITIAL_CASH

    if equity_curve.empty:
        raise ValueError("Window backtest produced no rows; check the date split.")

    running_peak = equity_curve.cummax()
    max_drawdown = (equity_curve / running_peak - 1).min()

    return {
        "final_portfolio_value": float(equity_curve.iloc[-1]),
        "total_return_pct": float((equity_curve.iloc[-1] / INITIAL_CASH - 1) * 100),
        "max_drawdown_pct": float(max_drawdown * 100),
        "equity_curve": equity_curve,
    }


def load_data():
    client = get_client(testnet=TESTNET)
    klines = client.get_historical_klines(SYMBOL, INTERVAL, START, END)
    raw_df = pd.DataFrame(
        klines,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )

    mvrv_series = load_mvrv_series(PROJECT_ROOT / "mvrv_historic.json")
    return build_feature_frame(raw_df, mvrv_series)


def split_masks(feat, validation_days):
    unique_dates = pd.Index(sorted(pd.unique(feat["date"])))
    if validation_days >= len(unique_dates):
        raise ValueError("validation_days must be smaller than the available data range.")

    validation_start = unique_dates[-validation_days]
    train_mask = feat["date"] < validation_start
    validation_mask = feat["date"] >= validation_start
    return train_mask, validation_mask, validation_start


def make_objective(feat, train_mask):
    def objective(trial):
        raw_weights = {key: trial.suggest_float(key, 0.0, 1.0) for key in WEIGHT_KEYS}
        weights = normalize_weights(raw_weights)
        result = backtest_window(feat, weights, train_mask)
        trial.set_user_attr("max_drawdown_pct", result["max_drawdown_pct"])
        return result["final_portfolio_value"]

    return objective


def parse_args():
    parser = argparse.ArgumentParser(description="Discover signal weights with Bayesian optimization.")
    parser.add_argument("--trials", type=int, default=75, help="Number of BO trials to run.")
    parser.add_argument(
        "--validation-days",
        type=int,
        default=365,
        help="Number of days to hold out at the end for the final score.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the sampler.")
    return parser.parse_args()


def main():
    args = parse_args()
    feat = load_data()
    train_mask, validation_mask, validation_start = split_masks(feat, args.validation_days)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(make_objective(feat, train_mask), n_trials=args.trials)

    best_weights = normalize_weights(study.best_trial.params)
    train_result = backtest_window(feat, best_weights, train_mask)
    validation_result = backtest_window(feat, best_weights, validation_mask)

    payload = {
        "validation_start": str(validation_start),
        "best_trial_value": study.best_value,
        "best_weights": best_weights,
        "train_final_portfolio_value": train_result["final_portfolio_value"],
        "train_total_return_pct": train_result["total_return_pct"],
        "train_max_drawdown_pct": train_result["max_drawdown_pct"],
        "validation_final_portfolio_value": validation_result["final_portfolio_value"],
        "validation_total_return_pct": validation_result["total_return_pct"],
        "validation_max_drawdown_pct": validation_result["max_drawdown_pct"],
    }

    BEST_WEIGHTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Best weights discovered:")
    for key, value in best_weights.items():
        print(f"  {key}: {value:.4f}")
    print(f"Training final portfolio value: ${train_result['final_portfolio_value']:.2f}")
    print(f"Validation final portfolio value: ${validation_result['final_portfolio_value']:.2f}")
    print(f"Saved results to {BEST_WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
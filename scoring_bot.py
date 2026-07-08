import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exchange.client import get_client


SYMBOL = "BTCUSDT"
INTERVAL = "1d"
START = "1 Jan 2018"
END = "1 Jan 2026"
TESTNET = False
INITIAL_CASH = 10000
FEE_RATE = 0.001
ZONE_EDGES = [-0.2, 0.0, 0.2, 0.4]
EXPOSURE_LEVELS = np.linspace(0, 1, len(ZONE_EDGES) + 1)
ASSET_DIR = PROJECT_ROOT / "docs" / "assets"


def scale_to_range(series, lower, upper):
    clipped = series.clip(lower=lower, upper=upper)
    return 2 * (clipped - lower) / (upper - lower) - 1


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(period).mean() / loss.rolling(period).mean().replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def load_mvrv_series(json_path):
    with json_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    mvrv_df = pd.DataFrame(records)

    if "d" not in mvrv_df.columns or "mvrvZscore" not in mvrv_df.columns:
        raise ValueError("mvrv_historic.json must contain 'd' and 'mvrvZscore' fields")

    mvrv_df["date"] = pd.to_datetime(mvrv_df["d"]).dt.date
    mvrv_df["mvrv"] = pd.to_numeric(mvrv_df["mvrvZscore"], errors="coerce")
    return mvrv_df[["date", "mvrv"]].dropna(subset=["mvrv"])


def build_feature_frame(raw_df, mvrv_series):
    feat = raw_df.copy()

    for column in ["open", "high", "low", "close", "volume", "taker_buy_base_asset_volume"]:
        feat[column] = pd.to_numeric(feat[column], errors="coerce")

    feat["open_time"] = pd.to_datetime(feat["open_time"], unit="ms")
    feat["date"] = feat["open_time"].dt.date

    feat["sma_200"] = feat["close"].rolling(200).mean()
    feat["sma_dist"] = (feat["close"] - feat["sma_200"]) / feat["sma_200"]
    feat["ema_50"] = feat["close"].ewm(span=50, adjust=False).mean()
    feat["ema_slope"] = feat["ema_50"].diff()
    feat["rsi_14"] = compute_rsi(feat["close"], period=14)
    feat["exnet"] = feat["taker_buy_base_asset_volume"] / (
        feat["volume"] - feat["taker_buy_base_asset_volume"] + 1e-9
    )
    feat["volatility_20"] = feat["close"].pct_change().rolling(20).std()

    feat = feat.merge(mvrv_series, on="date", how="left")

    feat["sma_dist_scaled"] = scale_to_range(feat["sma_dist"], -0.5, 1.0)
    feat["ema_slope_scaled"] = scale_to_range(feat["ema_slope"], -250, 250)
    feat["rsi_scaled"] = scale_to_range(feat["rsi_14"], 20, 80)
    feat["exnet_scaled"] = scale_to_range(feat["exnet"], 0.8, 1.2)
    feat["mvrv_scaled"] = scale_to_range(feat["mvrv"].fillna(0), -5.0, 5.0)
    feat["volatility_scaled"] = scale_to_range(feat["volatility_20"], 0.0, 0.06)

    feat["combined_signal"] = (
        0.001 * feat["sma_dist_scaled"]
        + 0.4882 * feat["ema_slope_scaled"]
        + 0.5020 * feat["rsi_scaled"]
        + 0.0007 * feat["exnet_scaled"]
        + 0.009 * feat["mvrv_scaled"]
    )

    feat["exposure"] = pd.cut(
        feat["combined_signal"],
        bins=[-np.inf] + ZONE_EDGES + [np.inf],
        labels=EXPOSURE_LEVELS,
        include_lowest=True,
    ).astype(float)

    feat["exposure_shifted"] = feat["exposure"].shift(1)
    feat["exposure_change"] = feat["exposure"] != feat["exposure_shifted"]
    feat["days_in_zone"] = feat.groupby(feat["exposure_change"].cumsum()).cumcount() + 1
    feat["final_exposure"] = np.where(feat["days_in_zone"] >= 3, feat["exposure"], feat["exposure_shifted"])

    feat["combined_signal_rolling"] = feat["combined_signal"].rolling(window=3).mean()
    feat["final_exposure_rolling"] = pd.cut(
        feat["combined_signal_rolling"],
        bins=[-np.inf] + ZONE_EDGES + [np.inf],
        labels=EXPOSURE_LEVELS,
        include_lowest=True,
    ).astype(float)

    feat["final_exposure_5d"] = np.where(feat["days_in_zone"] >= 5, feat["exposure"], feat["exposure_shifted"])
    feat["combined_signal_rolling_5"] = feat["combined_signal"].rolling(window=5).mean()
    feat["final_exposure_rolling_5"] = pd.cut(
        feat["combined_signal_rolling_5"],
        bins=[-np.inf] + ZONE_EDGES + [np.inf],
        labels=EXPOSURE_LEVELS,
        include_lowest=True,
    ).astype(float)

    return feat


def backtest_exposure(feat, exposure_col):
    position = feat[exposure_col].astype(float).fillna(0.0)
    position_change = position.diff().abs().fillna(0.0)
    transaction_cost = position_change * FEE_RATE
    strategy_return = position.shift(1).fillna(0.0) * feat["close"].pct_change().fillna(0.0) - transaction_cost
    equity_curve = (1 + strategy_return).cumprod() * INITIAL_CASH

    final_portfolio_value = equity_curve.iloc[-1]
    buy_and_hold_value = (feat["close"].iloc[-1] / feat["close"].iloc[0]) * INITIAL_CASH

    return {
        "final_portfolio_value": final_portfolio_value,
        "total_return_pct": (final_portfolio_value / INITIAL_CASH - 1) * 100,
        "buy_and_hold_value": buy_and_hold_value,
        "buy_and_hold_return_pct": (buy_and_hold_value / INITIAL_CASH - 1) * 100,
        "equity_curve": equity_curve,
    }


def print_results(results):
    for exposure_col, result in results.items():
        print(f"Exposure Column: {exposure_col}")
        print(f"Final Portfolio Value: ${result['final_portfolio_value']:.2f}")
        print(f"Total Return: {result['total_return_pct']:.2f}%")
        print(f"Buy and Hold Value: ${result['buy_and_hold_value']:.2f}")
        print(f"Buy and Hold Return: {result['buy_and_hold_return_pct']:.2f}%")
        print("-" * 40)


def save_overview_chart(feat):
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(feat["open_time"], feat["close"], label="Close Price", color="steelblue", linewidth=1.5)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("BTCUSDT Price", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.plot(feat["open_time"], feat["combined_signal"], label="Combined Signal", color="darkred", alpha=0.8)
    ax2.plot(feat["open_time"], feat["final_exposure_rolling_5"], label="Final Exposure", color="darkgreen", alpha=0.8)
    ax2.set_ylabel("Signal / Exposure", color="darkred")
    ax2.tick_params(axis="y", labelcolor="darkred")

    fig.suptitle("BTCUSDT Signal Overview")
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "signal_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_equity_curve_chart(feat, equity_curve):
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(feat["open_time"], feat["close"], label="Close Price", color="steelblue")
    ax.plot(feat["open_time"], equity_curve, label="Strategy Equity Curve", color="darkred")
    ax.set_title("BTCUSDT Close Price and Strategy Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "equity_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
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
    feat = build_feature_frame(raw_df, mvrv_series)

    strategy_names = ["final_exposure", "final_exposure_rolling", "final_exposure_5d", "final_exposure_rolling_5"]
    results = {name: backtest_exposure(feat, name) for name in strategy_names}

    print_results(results)
    save_overview_chart(feat)
    save_equity_curve_chart(feat, results["final_exposure_rolling_5"]["equity_curve"])


if __name__ == "__main__":
    main()
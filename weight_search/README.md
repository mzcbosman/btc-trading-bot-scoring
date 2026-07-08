# Weight Search

This folder contains a separate Bayesian-optimization entrypoint for discovering signal weights.

Run it after setting up your Binance credentials:

```bash
python3 weight_search/optimize_weights.py --trials 75 --validation-days 365
```

The script tunes the weight blend on the training history and reports a final holdout score on the last `validation-days` of data. The best run is written to `weight_search/best_weights.json`.
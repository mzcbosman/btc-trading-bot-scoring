from binance.client import Client
from config.config import API_KEY_Test, API_SECRET_Test, API_KEY, API_SECRET


def _require_value(value, name):
    if not value:
        raise RuntimeError(
            f"Missing {name}. Put it in a local .env file before running the bot."
        )
    return value


def get_client(testnet=True):
    if testnet:
        api_key = _require_value(API_KEY_Test, "BINANCE_API_KEY_TEST")
        api_secret = _require_value(API_SECRET_Test, "BINANCE_SECRET_KEY_TEST")
        client = Client(api_key, api_secret, testnet=True)
        client.API_URL = 'https://testnet.binance.vision/api'
    else:
        api_key = _require_value(API_KEY, "BINANCE_API_KEY")
        api_secret = _require_value(API_SECRET, "BINANCE_SECRET_KEY")
        client = Client(api_key, api_secret)
        client.API_URL = 'https://api.binance.com/api'

    return client
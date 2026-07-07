import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"


def _load_dotenv():
    if not ENV_FILE.exists():
        return

    for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()


def _get_env_var(name, default=""):
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else default


API_KEY_Test = _get_env_var("BINANCE_API_KEY_TEST")
API_SECRET_Test = _get_env_var("BINANCE_SECRET_KEY_TEST")
API_KEY = _get_env_var("BINANCE_API_KEY")
API_SECRET = _get_env_var("BINANCE_SECRET_KEY")

SYMBOL = "BTCUSDT"
TRADE_SIZE = 0.0001
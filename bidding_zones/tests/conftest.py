"""Put bidding_zones/ on sys.path so ``import bzgen`` works from any working directory
(including a pytest run from the repository root)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

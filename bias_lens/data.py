"""Load CrowS-Pairs (Nangia et al., 2020).

Contains offensive stereotypes by design. Never show random pairs on a
landing page; always put them behind the content warning.
"""
from pathlib import Path

import pandas as pd

URL = ("https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/"
       "data/crows_pairs_anonymized.csv")
CACHE = Path(__file__).resolve().parents[1] / "data" / "crows_pairs.csv"


def load_crows_pairs() -> pd.DataFrame:
    """Return one row per pair with stereo_sent / anti_sent resolved.

    In the raw file, `stereo_antistereo` says what `sent_more` is:
      'stereo'     -> sent_more is the stereotype
      'antistereo' -> sent_more is the anti-stereotype (so sent_less is the stereotype)
    This follows the official CrowS-Pairs metric.
    """
    if CACHE.exists():
        df = pd.read_csv(CACHE)
    else:
        df = pd.read_csv(URL)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CACHE, index=False)

    df = df[["sent_more", "sent_less", "stereo_antistereo", "bias_type"]].copy()
    is_stereo = df["stereo_antistereo"] == "stereo"
    df["stereo_sent"] = df["sent_more"].where(is_stereo, df["sent_less"])
    df["anti_sent"] = df["sent_less"].where(is_stereo, df["sent_more"])
    return df.reset_index(drop=True)

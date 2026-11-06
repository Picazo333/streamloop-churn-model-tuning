"""Load the official StreamLoop churn table.

The CSV is the public IBM telco churn file named in the project statement.
It is downloaded when the study runs and is not stored in git.
"""

import hashlib
import io
import time
import urllib.request

from streamloop import _threads  # noqa: F401

import pandas as pd

OFFICIAL_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
EXPECTED_ROWS = 7043
EXPECTED_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def assert_official_payload(payload):
    digest = sha256_bytes(payload)
    if digest != EXPECTED_SHA256:
        raise ValueError(
            "The downloaded file does not match the pinned IBM telco CSV. "
            f"expected sha256 {EXPECTED_SHA256}, got {digest}."
        )
    return digest


def validate_telco_frame(frame):
    columns = list(frame.columns)
    if columns != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected columns: {columns}")
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"Unexpected row count: {len(frame)} (expected {EXPECTED_ROWS})")
    return frame


def download_bytes(url, timeout=60, attempts=3):
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read()
        except Exception as exc:  # network errors vary by environment
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"Could not download the official dataset from {url}: {last_error}") from last_error


def load_telco(url=OFFICIAL_URL, timeout=60):
    """Download and validate the official CSV. Return the frame and file facts."""
    payload = download_bytes(url, timeout=timeout)
    digest = assert_official_payload(payload)
    frame = pd.read_csv(io.BytesIO(payload))
    validate_telco_frame(frame)
    blank_total_charges = int(frame["TotalCharges"].astype(str).str.strip().eq("").sum())
    meta = {
        "source": "official-url",
        "url": url,
        "sha256": digest,
        "bytes": len(payload),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "blank_total_charges": blank_total_charges,
    }
    return frame, meta


def split_features_target(frame):
    """Separate X and y. The label map is fixed; it learns nothing from rows."""
    if "Churn" not in frame.columns:
        raise ValueError("The dataset has no Churn column.")
    labels = frame["Churn"]
    unexpected = sorted(set(labels.dropna().unique()) - {"Yes", "No"})
    if unexpected or labels.isna().any():
        raise ValueError(f"Churn must be Yes or No. Found: {unexpected or 'missing values'}")
    y = (labels == "Yes").astype(int)
    x = frame.drop(columns=["Churn"])
    return x, y


def positive_rate(y):
    return float(y.mean())

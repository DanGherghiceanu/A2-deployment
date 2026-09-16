#!/usr/bin/env python3
"""Exercise a running deployment. Works against localhost or the public URL.

    python test_api.py                                  # http://localhost:8080
    python test_api.py https://xray-api-xxxx.run.app    # deployed service
    python test_api.py https://... --image path/to/xray.jpeg

We run this against Cloud Run and screenshot the output for the Step 5 "API test"
deliverable — it exercises the health check, both prediction paths, and the
error handling in one pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

# A pneumonia-positive chest film from the Kaggle test split, served raw from
# GitHub. Replace with any public image URL if this ever goes stale.
SAMPLE_URL = (
    "https://raw.githubusercontent.com/DanGherghiceanu/A2-deployment/main/samples/xray-pneumonia.jpeg"
)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append(ok)
    print(f"  [{PASS if ok else FAIL}] {name}")
    if detail:
        for line in detail.splitlines():
            print(f"         {line}")


def test_health(base: str) -> None:
    print("\nHealth check")
    try:
        r = requests.get(f"{base}/health", timeout=30)
        body = r.json()
        check("GET /health returns 200", r.status_code == 200, json.dumps(body))
        check("model is loaded", body.get("model_loaded") is True)
    except Exception as exc:
        check("GET /health", False, str(exc))


def test_metadata(base: str) -> None:
    print("\nMetadata")
    try:
        r = requests.get(f"{base}/metadata", timeout=30)
        body = r.json()
        check("GET /metadata returns 200", r.status_code == 200)
        check(
            "classes and threshold present",
            body.get("classes") == ["NORMAL", "PNEUMONIA"] and "threshold" in body,
            f"threshold={body.get('threshold')}  version={body.get('model_version')}",
        )
    except Exception as exc:
        check("GET /metadata", False, str(exc))


def test_predict_url(base: str, url: str) -> None:
    print("\nPrediction by URL — POST /predict")
    print(f'  request body: {{"image_url": "{url[:60]}..."}}')
    try:
        r = requests.post(f"{base}/predict", json={"image_url": url}, timeout=120)
        body = r.json()
        if r.status_code != 200:
            check("returns 200", False, json.dumps(body))
            return
        check("returns 200", True)
        check(
            "prediction is a valid class",
            body.get("prediction") in {"NORMAL", "PNEUMONIA"},
            f"prediction={body['prediction']}  "
            f"P(pneumonia)={body['probability_pneumonia']:.4f}  "
            f"{body['inference_ms']:.0f} ms",
        )
        p = body.get("probability_pneumonia")
        check("probability is in [0, 1]", isinstance(p, float) and 0.0 <= p <= 1.0)
        check(
            "label agrees with threshold",
            body["label_index"] == int(p >= body["threshold"]),
        )
    except Exception as exc:
        check("POST /predict", False, str(exc))


def test_predict_upload(base: str, image_path: Path | None, url: str) -> None:
    print("\nPrediction by upload — POST /predict/file")
    try:
        if image_path:
            raw = image_path.read_bytes()
            name = image_path.name
        else:
            raw = requests.get(url, timeout=60, headers={"User-Agent": "xray-api-test/1.0"}).content
            name = "sample.jpeg"

        r = requests.post(
            f"{base}/predict/file",
            files={"file": (name, raw, "image/jpeg")},
            timeout=120,
        )
        body = r.json()
        if r.status_code != 200:
            check("returns 200", False, json.dumps(body))
            return
        check(
            "returns 200 with a prediction",
            body.get("prediction") in {"NORMAL", "PNEUMONIA"},
            f"file={name}  prediction={body['prediction']}  "
            f"P(pneumonia)={body['probability_pneumonia']:.4f}",
        )
    except Exception as exc:
        check("POST /predict/file", False, str(exc))


def test_errors(base: str) -> None:
    print("\nError handling")
    cases = [
        ("missing image_url", {"json": {}}, 422),
        ("non-http scheme", {"json": {"image_url": "file:///etc/passwd"}}, 400),
        ("internal address", {"json": {"image_url": "http://169.254.169.254/"}}, 400),
        ("unreachable host", {"json": {"image_url": "https://not-a-real-host.invalid/x.jpg"}}, 400),
    ]
    for name, kwargs, expected in cases:
        try:
            r = requests.post(f"{base}/predict", timeout=60, **kwargs)
            check(
                f"{name} rejected with {expected}",
                r.status_code == expected,
                f"got {r.status_code}: {str(r.json().get('detail'))[:90]}",
            )
        except Exception as exc:
            check(name, False, str(exc))


def test_ui(base: str) -> None:
    print("\nWeb UI")
    try:
        r = requests.get(base + "/", timeout=30)
        check(
            "GET / serves the page",
            r.status_code == 200 and "text/html" in r.headers.get("content-type", ""),
        )
        check("GET /docs serves API docs", requests.get(f"{base}/docs", timeout=30).status_code == 200)
    except Exception as exc:
        check("web UI", False, str(exc))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url", nargs="?", default="http://localhost:8080")
    parser.add_argument("--image", type=Path, help="Local X-ray to upload instead of the sample")
    parser.add_argument("--url", default=SAMPLE_URL, help="Image URL to test /predict with")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    print(f"Testing {base}")

    test_health(base)
    test_metadata(base)
    test_predict_url(base, args.url)
    test_predict_upload(base, args.image, args.url)
    test_errors(base)
    test_ui(base)

    passed, total = sum(results), len(results)
    print(f"\n{passed}/{total} checks passed\n")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

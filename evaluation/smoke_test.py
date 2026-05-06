from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

from evaluation.metrics import summarize_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://localhost:8001")
    parser.add_argument("--video", required=True)
    parser.add_argument("--context", default="pitch")
    args = parser.parse_args()

    with open(args.video, "rb") as file_obj:
        response = requests.post(
            f"{args.api_base}/api/v1/sessions/analyse",
            files={"video": (Path(args.video).name, file_obj, "video/mp4")},
            data={"context": args.context},
            timeout=60,
        )
    response.raise_for_status()
    session_id = response.json()["session_id"]

    status = "queued"
    while status not in {"completed", "failed"}:
        time.sleep(2)
        status_resp = requests.get(f"{args.api_base}/api/v1/sessions/{session_id}/status", timeout=30)
        status_resp.raise_for_status()
        status = status_resp.json()["status"]

    result_resp = requests.get(f"{args.api_base}/api/v1/sessions/{session_id}/result", timeout=30)
    result_resp.raise_for_status()
    payload = result_resp.json()
    metrics = summarize_result(payload["result"])
    print(json.dumps({"session_id": session_id, "status": status, "metrics": metrics.__dict__}, indent=2))


if __name__ == "__main__":
    main()

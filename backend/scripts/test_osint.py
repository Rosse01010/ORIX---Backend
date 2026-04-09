#!/usr/bin/env python3
"""
test_osint.py
─────────────
Smoke test for the ORIX OSINT subsystem.

Usage:
    # Test with a random embedding (no image needed):
    python scripts/test_osint.py --api-url http://localhost:8000

    # Test with an image (extracts embedding server-side):
    python scripts/test_osint.py --image test_face.jpg --api-url http://localhost:8000

    # Test enrichment by face_id:
    python scripts/test_osint.py --face-id <uuid> --api-url http://localhost:8000

Requires: OSINT_ENABLED=true on the target API server.
"""
from __future__ import annotations

import argparse
import json
import sys

import httpx
import numpy as np


def _random_embedding() -> list:
    """Generate a random L2-normalised 512-dim embedding for testing."""
    vec = np.random.randn(512).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="ORIX OSINT smoke test")
    parser.add_argument("--api-url", default="http://localhost:8000",
                        help="Base URL of the ORIX API")
    parser.add_argument("--image", default=None,
                        help="Path to a face image for enrich-face test")
    parser.add_argument("--face-id", default=None,
                        help="Existing person UUID for enrich-face test")
    args = parser.parse_args()

    base = args.api_url.rstrip("/")
    client = httpx.Client(timeout=30)

    # ── 1. Health check ───────────────────────────────────────────────
    print("[1/4] Checking OSINT health...")
    try:
        r = client.get(f"{base}/api/osint/health")
        if r.status_code == 403:
            print("      OSINT is DISABLED on this server.")
            print("      Set OSINT_ENABLED=true and restart.")
            sys.exit(1)
        r.raise_for_status()
        health = r.json()
        print(f"      Status:    {health['status']}")
        print(f"      Providers: {json.dumps(health['providers'], indent=2)}")
    except httpx.ConnectError:
        print(f"      Cannot connect to {base}")
        sys.exit(1)

    # ── 2. Search by embedding ────────────────────────────────────────
    print("\n[2/4] Testing POST /api/osint/search with random embedding...")
    embedding = _random_embedding()
    r = client.post(
        f"{base}/api/osint/search",
        json={"embedding": embedding, "top_k": 5},
    )
    r.raise_for_status()
    report = r.json()
    query_id = report["query_id"]
    print(f"      Query ID:        {query_id}")
    print(f"      Matches:         {len(report['matches'])}")
    print(f"      Risk score:      {report['risk_score']}")
    print(f"      Providers:       {report['providers_queried']}")
    print(f"      Processing time: {report['processing_time_ms']} ms")

    for i, m in enumerate(report["matches"][:3]):
        print(f"      Match #{i}: source={m['source']}, "
              f"confidence={m['confidence']}, id={m['external_id']}")

    # ── 3. Retrieve cached report ─────────────────────────────────────
    print(f"\n[3/4] Testing GET /api/osint/report/{query_id}...")
    r = client.get(f"{base}/api/osint/report/{query_id}")
    r.raise_for_status()
    cached = r.json()
    print(f"      Cached:  {cached.get('cached', False)}")
    print(f"      Matches: {len(cached['matches'])}")

    # ── 4. Enrich face (optional) ─────────────────────────────────────
    if args.image:
        print(f"\n[4/4] Testing POST /api/osint/enrich-face with image: {args.image}")
        with open(args.image, "rb") as f:
            r = client.post(
                f"{base}/api/osint/enrich-face",
                files={"file": (args.image, f, "image/jpeg")},
                data={"top_k": "5"},
            )
        r.raise_for_status()
        enriched = r.json()
        print(f"      Query ID:   {enriched['query_id']}")
        print(f"      Matches:    {len(enriched['matches'])}")
        print(f"      Risk score: {enriched['risk_score']}")
    elif args.face_id:
        print(f"\n[4/4] Testing POST /api/osint/enrich-face with face_id: {args.face_id}")
        r = client.post(
            f"{base}/api/osint/enrich-face",
            data={"face_id": args.face_id, "top_k": "5"},
        )
        r.raise_for_status()
        enriched = r.json()
        print(f"      Query ID:   {enriched['query_id']}")
        print(f"      Matches:    {len(enriched['matches'])}")
        print(f"      Risk score: {enriched['risk_score']}")
    else:
        print("\n[4/4] Skipping enrich-face test (no --image or --face-id provided)")

    print("\nAll OSINT tests passed.")


if __name__ == "__main__":
    main()

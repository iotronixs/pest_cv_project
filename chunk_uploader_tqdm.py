#!/usr/bin/env python3
"""
chunk_uploader_tqdm.py
Split a large base64 text file and send chunks to the Flask endpoint:
  POST {server_url}?id={upload_id}&idx={i}&total={total}
Body: raw base64 text chunk

Usage:
  pip install requests tqdm
  python chunk_uploader_tqdm.py --file base64.txt --server http://10.202.207.165:5000/api/upload_b64_chunk --chunk 50000
"""

import argparse
import requests
import uuid
import time
import sys
import os
from tqdm import tqdm

def send_chunk(server_url, upload_id, idx, total, chunk_text, max_retries=3, timeout=30):
    params = {"id": upload_id, "idx": str(idx), "total": str(total)}
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(server_url, params=params, data=chunk_text.encode("utf-8"), timeout=timeout)
            if resp.status_code == 200:
                try:
                    return True, resp.json()
                except Exception:
                    return True, {"text": resp.text}
            else:
                print(f"[WARN] server returned {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            print(f"[WARN] exception sending chunk {idx}: {e}")
        time.sleep(1.5 * attempt)
    return False, {"error": "max_retries_exceeded"}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", "-f", required=True, help="Path to base64 text file (plain text)")
    p.add_argument("--server", "-s", required=True, help="Server endpoint, e.g. http://10.202.207.165:5000/api/upload_b64_chunk")
    p.add_argument("--chunk", "-c", type=int, default=50000, help="Chunk size (characters). Try 50000 or 80000.")
    p.add_argument("--id", help="Optional upload id (default: random hex)")
    p.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds")
    args = p.parse_args()

    if not os.path.isfile(args.file):
        print("File not found:", args.file)
        sys.exit(1)

    with open(args.file, "r", encoding="utf-8") as fh:
        text = fh.read()

    # Remove whitespace/newlines
    text = "".join(text.split())

    total_len = len(text)
    if total_len == 0:
        print("Empty input file.")
        sys.exit(1)

    chunk_size = args.chunk
    total_chunks = (total_len + chunk_size - 1) // chunk_size
    upload_id = args.id or uuid.uuid4().hex[:12]

    print(f"Uploading file: {args.file}")
    print(f"Length: {total_len} chars  chunk_size: {chunk_size}  total_chunks: {total_chunks}")
    print(f"Server: {args.server}")
    print(f"Upload ID: {upload_id}")
    print("Starting upload...")

    # tqdm progress bar over chunks
    with tqdm(total=total_chunks, desc="Uploading chunks", unit="chunk") as pbar:
        for i in range(total_chunks):
            idx = i + 1
            start = i * chunk_size
            chunk_text = text[start:start + chunk_size]
            ok, info = send_chunk(args.server, upload_id, idx, total_chunks, chunk_text, timeout=args.timeout)
            if not ok:
                print(f"\n[ERROR] Failed to upload chunk {idx}/{total_chunks}. Aborting.")
                print("Server response:", info)
                sys.exit(2)
            else:
                pbar.update(1)
                # If last chunk, capture final info
                if idx == total_chunks:
                    if isinstance(info, dict) and info.get("img_url"):
                        print("\nUpload complete. Server returned image URL:", info["img_url"])
                    else:
                        print("\nUpload complete. Server response:", info)

    print("All done.")

if __name__ == "__main__":
    main()

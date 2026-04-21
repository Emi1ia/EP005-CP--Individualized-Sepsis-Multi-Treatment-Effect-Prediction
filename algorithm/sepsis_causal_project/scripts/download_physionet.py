#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urljoin

import requests


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.links.append(href)


def normalize_base_url(url: str) -> str:
    if not url.startswith("http://") and not url.startswith("https://"):
        raise ValueError(f"Invalid URL: {url}")
    if not url.endswith("/"):
        url += "/"
    return url


def rel_to_local_path(dest: Path, rel: str) -> Path:
    rel = unquote(rel)
    posix = PurePosixPath(rel)
    if posix.is_absolute() or ".." in posix.parts:
        raise ValueError(f"Unsafe relative path: {rel}")
    return dest.joinpath(*posix.parts)


def crawl_files(
    session: requests.Session,
    base_url: str,
    timeout_sec: int,
    retries: int,
) -> list[tuple[str, str]]:
    queue: deque[str] = deque([base_url])
    visited: set[str] = set()
    files: list[tuple[str, str]] = []

    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)

        r = request_with_retry(
            session=session,
            url=current,
            timeout_sec=timeout_sec,
            retries=retries,
            stream=False,
        )
        if r.status_code == 401:
            raise PermissionError(
                f"Unauthorized for {current}. Check PhysioNet credentials and approved access."
            )
        r.raise_for_status()
        parser = LinkParser()
        parser.feed(r.text)

        for href in parser.links:
            if href.startswith("#"):
                continue
            if href.startswith("mailto:"):
                continue
            if href == "../":
                continue
            # Skip URL parameters links such as "download" helpers.
            if "?" in href:
                continue

            abs_url = urljoin(current, href)
            if not abs_url.startswith(base_url):
                continue

            rel = abs_url[len(base_url) :]
            if not rel:
                continue

            if href.endswith("/"):
                queue.append(abs_url)
            else:
                files.append((rel, abs_url))

    # Keep deterministic order.
    files = sorted(set(files), key=lambda x: x[0])
    return files


def should_download(
    rel_path: str,
    include_filters: list[str],
    exclude_filters: list[str],
) -> bool:
    if include_filters and not any(f in rel_path for f in include_filters):
        return False
    if exclude_filters and any(f in rel_path for f in exclude_filters):
        return False
    return True


def request_with_retry(
    session: requests.Session,
    url: str,
    timeout_sec: int,
    retries: int,
    stream: bool,
) -> requests.Response:
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, stream=stream, timeout=(20, timeout_sec))
            return r
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < retries:
                sleep_sec = min(2**attempt, 20)
                print(f"[retry] {url} attempt={attempt}/{retries} sleep={sleep_sec}s", file=sys.stderr)
                time.sleep(sleep_sec)
            else:
                break
    raise RuntimeError(f"request failed after {retries} attempts: {url} :: {last_err}")


def download_file(
    session: requests.Session,
    file_url: str,
    local_path: Path,
    timeout_sec: int,
    retries: int,
) -> str:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = local_path.with_suffix(local_path.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    with request_with_retry(
        session=session,
        url=file_url,
        timeout_sec=timeout_sec,
        retries=retries,
        stream=True,
    ) as r:
        if r.status_code == 401:
            raise PermissionError(
                f"Unauthorized for {file_url}. Check PhysioNet credentials and approved access."
            )
        r.raise_for_status()
        content_len = r.headers.get("Content-Length")
        expected = int(content_len) if content_len and content_len.isdigit() else None

        if local_path.exists() and expected is not None and local_path.stat().st_size == expected:
            return "skip_same_size"

        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    os.replace(tmp, local_path)
    return "downloaded"


def main() -> int:
    ap = argparse.ArgumentParser(description="Recursive PhysioNet downloader for Windows/Linux.")
    ap.add_argument("--base-url", required=True, help="PhysioNet /files/ URL ending in dataset version folder.")
    ap.add_argument("--dest", required=True, help="Local destination folder.")
    ap.add_argument("--username", default=os.environ.get("PHYSIONET_USERNAME"))
    ap.add_argument("--password", default=os.environ.get("PHYSIONET_PASSWORD"))
    ap.add_argument("--include", action="append", default=[], help="Substring filter; can pass multiple times.")
    ap.add_argument("--exclude", action="append", default=[], help="Substring filter; can pass multiple times.")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--timeout-sec", type=int, default=120)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base_url = normalize_base_url(args.base_url)
    dest = Path(args.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    if args.username:
        if not args.password:
            print("Error: username provided but password missing. Set --password or PHYSIONET_PASSWORD.", file=sys.stderr)
            return 2
        session.auth = (args.username, args.password)

    print(f"[crawl] {base_url}")
    files = crawl_files(
        session=session,
        base_url=base_url,
        timeout_sec=args.timeout_sec,
        retries=args.retries,
    )
    files = [
        (rel, url)
        for rel, url in files
        if should_download(rel, include_filters=args.include, exclude_filters=args.exclude)
    ]
    if args.max_files is not None:
        files = files[: args.max_files]

    print(f"[plan] files={len(files)} dest={dest}")

    downloaded = 0
    skipped = 0
    failed = 0

    for i, (rel, url) in enumerate(files, 1):
        local_path = rel_to_local_path(dest, rel)
        if local_path.exists():
            skipped += 1
            if i % 5000 == 0 or i == len(files):
                print(f"[exists] {i}/{len(files)} {rel}")
            continue
        if args.dry_run:
            print(f"[dry-run] {i}/{len(files)} {rel}")
            continue
        try:
            status = download_file(
                session=session,
                file_url=url,
                local_path=local_path,
                timeout_sec=args.timeout_sec,
                retries=args.retries,
            )
            if status == "downloaded":
                downloaded += 1
            else:
                skipped += 1
            print(f"[{status}] {i}/{len(files)} {rel}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[failed] {i}/{len(files)} {rel} :: {e}", file=sys.stderr)

    print(f"[done] downloaded={downloaded} skipped={skipped} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

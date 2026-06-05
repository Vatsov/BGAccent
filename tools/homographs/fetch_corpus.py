"""Polite chitanka downloader: search by title, pick a real (large enough)
edition, download .txt.zip, extract into data/corpus. Sleeps between requests."""

from __future__ import annotations

import io
import re
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15)"
OUT = Path("data/corpus")
MIN_WORDS = 20000
HAVE = {"3753", "2170", "2321", "10588", "1207", "7153", "7944"}

QUERIES = [
    # Bulgarian originals
    "Тютюн",
    "Време разделно",
    "Железният светилник",
    "Бай Ганьо",
    "Крадецът на праскови",
    "Хайка за вълци",
    "Гераците",
    "Записки по българските въстания",
    "Иван Кондарев",
    # World classics (translated)
    "Война и мир",
    "Ана Каренина",
    "Братя Карамазови",
    "Идиот Достоевски",
    "Клетниците",
    "Тримата мускетари",
    "Дон Кихот",
    "Двадесет хиляди левги под водата",
    "Островът на съкровищата",
    "Джейн Еър",
    "Брулени хълмове",
    "Парфюмът",
    "Чумата Камю",
    "Шерлок Холмс",
    "Граф Дракула",
    # Modern / genre (audiobook register)
    "Хари Потър философският камък",
    "Игра на тронове",
    "Кодът на Да Винчи",
    "Сияние Кинг",
    "Малкият принц",
    "1984 Оруел",
]


def fetch(url: str, binary: bool = False):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=90) as r:
        data = r.read()
    return data if binary else data.decode("utf-8", "replace")


def candidates(query: str) -> list[str]:
    enc = urllib.parse.quote(query)
    html = fetch(f"https://chitanka.info/search?q={enc}")
    seen, out = set(), []
    for m in re.finditer(r"/text/(\d+)-[a-z0-9-]+", html):
        slug = m.group(0)
        tid = m.group(1)
        if slug not in seen:
            seen.add(slug)
            out.append((tid, slug))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ok = 0
    for q in QUERIES:
        try:
            cands = candidates(q)
        except Exception as e:
            print(f"❌ search err [{q}]: {e}", flush=True)
            time.sleep(5)
            continue
        time.sleep(4)
        picked = False
        for tid, slug in cands[:3]:
            if tid in HAVE:
                picked = True
                print(f"⏭  вече имам [{q}] id={tid}", flush=True)
                break
            try:
                blob = fetch(f"https://chitanka.info{slug}.txt.zip", binary=True)
                zf = zipfile.ZipFile(io.BytesIO(blob))
                name = zf.namelist()[0]
                raw = zf.read(name).decode("utf-8", "replace")
                wc = len(raw.split())
            except Exception as e:
                print(f"   skip cand {tid} [{q}]: {e}", flush=True)
                time.sleep(5)
                continue
            if wc >= MIN_WORDS:
                (OUT / name).write_text(raw, encoding="utf-8")
                HAVE.add(tid)
                ok += 1
                picked = True
                print(f"✅ {q:40} id={tid:6} {wc:7,} думи", flush=True)
                time.sleep(6)
                break
            else:
                print(f"   твърде малък cand {tid} [{q}] ({wc} думи), пробвам следващ", flush=True)
                time.sleep(5)
        if not picked:
            print(f"❌ няма подходящ резултат: {q}", flush=True)
        time.sleep(2)
    print(f"\n=== свалени нови: {ok} ===", flush=True)


if __name__ == "__main__":
    main()

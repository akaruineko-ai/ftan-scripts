"""Step 12: Build the H/H2/HR multi-label hate dataset (arrow).

Reads the pseudo-labeled multi-labeled JSONL produced by `KoalaAI/Text-Moderation`
(`data/exps/multi-labeled/fantastic_offensive_moderated.jsonl`), drops the
untrustworthy teacher `moderation_results`, and derives a deterministic multi-label
in {H, H2, HR} from regex + the FTAN offensive confidence stored in `origin_label`:

    * H  (hate)            <- slur / group-hate lexicon (reddit_vocab._SLURS + _GROUP_HOSTILE)
    * HR (harassment)      <- 2nd-person targeted insult/attack (reddit_vocab "attack")
    * H2 (hate/threatening)<- H  AND a serious-harm / threat term

Decision policy (per row):
    1. `ftan_<conf>` with conf > min_conf AND regex matches -> keep (regex labels).
    2. otherwise (not ftan, or ftan<=min_conf): regex matches -> keep (regex labels).
    3. FTAN borderline: ftan conf in (ftan_borderline_lo, ftan_borderline_hi]
       AND top moderation confidence > ftan_borderline_mod_threshold AND regex
       misses -> keep as ['H'] (hate assumed from FTAN + teacher agreement).
    4. otherwise regex misses -> fallback to H ONLY when `origin_label` does NOT
       parse to ftan AND the top moderation confidence is < 0.7.

Improvements over basic regex:
    - Obfuscation-tolerant patterns (separators, repeated letters, homoglyphs,
      fullwidth, leet) via normalize_full().
    - Censored-consonant detection (n****r, f*ck) via first/last letter matching.
    - Dual-path detection (original + normalized text union).
    - Embedded-slur negative filter (excludes niggardly etc).
    - Expanded lexicon: _GROUP_HOSTILE covers religion, race, dehumanization,
      LGBTQ+, political/ideological hate.

Output is a Hugging Face `DatasetDict` (arrow) mirroring `06_export.py`:

    data/exps/final/dataset   (train, test, ...)
    data/exps/final/stats.json
    data/exps/final/dataset_card.md

Usage:
    python scripts/multi_label_dataset.py                 # full run -> data/exps/final
    python scripts/multi_label_dataset.py --limit 5000 --out /tmp/final_test
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc

# make `reddit_vocab` importable whether run from repo root or scripts/
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from reddit_vocab import (  # noqa: E402
    classify_regex,
    normalize,
    normalize_full,
    SLUR_OBF_RES,
    THREAT_OBF_RES,
    _is_embedded_slur,
    _matches_censored_slur,
)

# --------------------------------------------------------------------------- #
# Regex engine (obfuscation-tolerant)
# --------------------------------------------------------------------------- #
# All matching below runs on `normalize(text)`, which reverses fullwidth /
# homoglyph / leet mutations; the `*_OBF_RES` patterns additionally tolerate
# separators, repeated letters and substring (embedded) matches.


def parse_ftan_conf(origin_label) -> float | None:
    """Return FTAN offensive confidence, or None if not a `ftan_<conf>` label."""
    if not isinstance(origin_label, str) or not origin_label.startswith("ftan_"):
        return None
    try:
        return float(origin_label[len("ftan_"):])
    except ValueError:
        return None


def _any_match(text: str, patterns) -> list:
    """Return list of (pattern, match) tuples for all patterns that match text."""
    results = []
    for p in patterns:
        m = p.search(text)
        if m:
            results.append((p, m))
    return results


def derive_labels(text: str) -> list[str]:
    """Return the multi-label set (ordered H, HR, H2) for one text, or [].

    Dual-path detection:
      1. Run patterns on normalize_full(text) — catches obfuscated forms.
      2. Run patterns on raw text — catches cases where normalization
         inadvertently strips a feature (e.g. a leet digit the strict pattern
         already handles).
    Union of both paths is taken.

    Embedded-slur filter: if a slur match is embedded inside a longer word
    (e.g. 'nigger' inside 'niggardly'), it is excluded.
    """
    hate = False
    threat = False

    # Path 1: normalized + de-censored text (primary)
    n = normalize_full(text)
    n_pre_de = normalize(text)  # before de-censoring (keeps *, $, @, #)
    for _, m in _any_match(n, SLUR_OBF_RES):
        if not _is_embedded_slur(m.start(), m.end(), n):
            hate = True
            break
    if not hate:
        # Path 2: original text (fallback for edge cases)
        for _, m in _any_match(text, SLUR_OBF_RES):
            if not _is_embedded_slur(m.start(), m.end(), text):
                hate = True
                break
    if not hate:
        # Path 3: censored-consonant check (n****r, f*ck, b*tch)
        # Must run on pre-de-censored text where *, $, @, # are still present
        hate = _matches_censored_slur(n_pre_de)

    # Threat detection on normalized text (word-boundary enforced)
    threat = _any_match(n, THREAT_OBF_RES)

    # Harassment: classify_regex on normalized text (improved _strict_pattern
    # with ch+ tolerates repeats like 'biiitch')
    harass = classify_regex(n)["category"] == "attack"

    labels: list[str] = []
    if hate:
        labels.append("H")
    if harass:
        labels.append("HR")
    if hate and threat:
        labels.append("H2")
    return labels


def top_moderation_conf(moderation_results) -> float | None:
    """Highest probability among the teacher `moderation_results`, or None."""
    if not moderation_results:
        return None
    best = None
    for m in moderation_results:
        p = m.get("probability") if isinstance(m, dict) else None
        if p is None:
            continue
        if best is None or p > best:
            best = p
    return best


def decide(
    text: str,
    origin_label,
    moderation_results,
    min_conf: float = 0.8,
    ftan_borderline_lo: float = 0.6,
    ftan_borderline_hi: float = 0.8,
    ftan_borderline_mod_threshold: float = 0.7,
) -> tuple[bool, list[str], str]:
    """Return (keep, labels, reason) using the layered confidence policy.

    1. `ftan_<conf>` with conf > min_conf AND regex matches -> keep (regex labels).
    2. otherwise (not ftan, or ftan<=min_conf): regex matches -> keep (regex labels).
    3. FTAN borderline: ftan conf in (ftan_borderline_lo, ftan_borderline_hi]
       AND top moderation confidence > ftan_borderline_mod_threshold AND regex
       misses -> keep as ['H'] (hate assumed from FTAN + teacher agreement).
    4. otherwise regex misses -> fallback to H ONLY when `origin_label` does NOT
       parse to ftan AND the top moderation confidence is < 0.7.
    """
    conf = parse_ftan_conf(origin_label)
    labels = derive_labels(text)

    if conf is not None and conf > min_conf:
        if labels:
            return True, labels, "ftan>0.8+regex"
        return False, [], "ftan>0.8 no-regex"

    if labels:
        return True, labels, "regex-only"

    # FTAN borderline override: FTAN moderate-confidence + teacher agrees
    top = top_moderation_conf(moderation_results)
    if (conf is not None
            and ftan_borderline_lo < conf <= ftan_borderline_hi
            and top is not None
            and top > ftan_borderline_mod_threshold):
        return True, ["H"], "ftan-borderline"

    # regex could not match -> last-resort fallback to H
    if conf is None and top is not None and top < 0.7:
        return True, ["H"], "fallback-H"
    return False, [], "dropped"


# --------------------------------------------------------------------------- #
# Arrow writing
# --------------------------------------------------------------------------- #
SCHEMA = pa.schema([
    ("text", pa.string()),
    ("labels", pa.list_(pa.string())),
    ("source", pa.string()),
    ("origin_label", pa.string()),
    ("split_origin", pa.string()),
    ("mutated", pa.int8()),
    ("variant", pa.int8()),
    ("cluster", pa.int64()),
])


def rows_to_table(rows: list[dict]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=SCHEMA)


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(SCRIPT_DIR.parent / "data/exps/multi-labeled/"
                                                  "fantastic_offensive_moderated.jsonl"))
    ap.add_argument("--out", default=str(SCRIPT_DIR.parent / "data/exps/final"))
    ap.add_argument("--min_conf", type=float, default=0.8,
                    help="keep rows with ftan confidence strictly above this")
    ap.add_argument("--ftan_borderline_lo", type=float, default=0.6,
                    help="lower bound for FTAN borderline override (exclusive)")
    ap.add_argument("--ftan_borderline_hi", type=float, default=0.8,
                    help="upper bound for FTAN borderline override (inclusive)")
    ap.add_argument("--ftan_borderline_mod_threshold", type=float, default=0.7,
                    help="min top moderation confidence for FTAN borderline override")
    ap.add_argument("--limit", type=int, default=None,
                    help="process at most N rows (small-data test mode)")
    ap.add_argument("--chunk", type=int, default=100_000,
                    help="rows per in-flight arrow chunk (memory bound)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # per-split accumulation of chunked arrow tables (never hold everything at once)
    split_tables: dict[str, list[pa.Table]] = {}
    split_buffers: dict[str, list[dict]] = {}
    split_kept: Counter = Counter()
    split_dropped: Counter = Counter()
    label_combo: Counter = Counter()
    label_hits: Counter = Counter()
    reasons: Counter = Counter()

    def flush(split: str):
        buf = split_buffers.get(split)
        if buf:
            split_tables.setdefault(split, []).append(rows_to_table(buf))
            split_buffers[split] = []

    from tqdm import tqdm

    seen = 0
    with open(args.input, "r", encoding="utf-8") as fh:
        pbar = tqdm(fh, total=args.limit, desc="labeling", unit="row")
        for line in pbar:
            if args.limit is not None and seen >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            seen += 1

            d = json.loads(line)
            keep, labels, reason = decide(
                d.get("text", ""), d.get("origin_label"), d.get("moderation_results"),
                min_conf=args.min_conf,
                ftan_borderline_lo=args.ftan_borderline_lo,
                ftan_borderline_hi=args.ftan_borderline_hi,
                ftan_borderline_mod_threshold=args.ftan_borderline_mod_threshold,
            )
            reasons[reason] += 1

            if not keep:
                split_dropped[d.get("split_origin", "?")] += 1
                pbar.set_postfix(kept=sum(split_kept.values()),
                                 dropped=sum(split_dropped.values()))
                continue

            split = d.get("split_origin", "train")
            row = {
                "text": d.get("text", ""),
                "labels": labels,
                "source": d.get("source", ""),
                "origin_label": d.get("origin_label", ""),
                "split_origin": split,
                "mutated": int(d.get("mutated", 0) or 0),
                "variant": int(d.get("variant", 0) or 0),
                "cluster": int(d.get("cluster", 0) or 0),
            }
            split_buffers.setdefault(split, []).append(row)
            if len(split_buffers[split]) >= args.chunk:
                flush(split)

            split_kept[split] += 1
            label_combo[tuple(labels)] += 1
            for lab in labels:
                label_hits[lab] += 1
            pbar.set_postfix(kept=sum(split_kept.values()),
                             dropped=sum(split_dropped.values()))

    for split in list(split_buffers.keys()):
        flush(split)

    # build DatasetDict
    from datasets import Dataset, DatasetDict

    ds_splits = {}
    for split, tables in split_tables.items():
        if not tables:
            continue
        combined = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
        ds_splits[split] = Dataset.from_dict(combined.to_pydict())

    dsd = DatasetDict(ds_splits)
    dsd.save_to_disk(str(out_dir / "dataset"))

    stats = {
        "input_rows_seen": seen,
        "min_conf": args.min_conf,
        "splits": {k: len(v) for k, v in dsd.items()},
        "kept_per_split": dict(split_kept),
        "dropped_per_split": dict(split_dropped),
        "total_kept": sum(split_kept.values()),
        "total_dropped": sum(split_dropped.values()),
        "decision_reasons": dict(reasons),
        "label_hits": dict(label_hits),
        "label_combos": {"+".join(c): n for c, n in label_combo.most_common()},
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    _write_card(out_dir, stats)

    print(json.dumps(stats, indent=2))


def _write_card(out_dir: Path, stats: dict):
    lines = [
        "---",
        "license: cc0-1.0",
        "task_categories:",
        "  - text-classification",
        "language:",
        "  - en",
        "size_categories:",
        "  - 10K<n<100K" if stats["total_kept"] < 100_000 else "  - 100K<n<1M",
        "---",
        "",
        "# ftan-2.0 hate / harassment multi-label dataset (H, H2, HR)",
        "",
        "Derived from the pseudo-labeled moderation JSONL by **dropping the teacher"
        "`moderation_results`** and instead assigning a deterministic multi-label in"
        "{H, H2, HR} from regex + the FTAN offensive confidence (`origin_label`):",
        "",
        "- `H`  (hate)             — slur / group-hate lexicon",
        "- `HR` (harassment)       — 2nd-person targeted insult / attack",
        "- `H2` (hate/threatening) — `H` AND a serious-harm / threat term",
        "",
        "Keep policy: rows with `ftan_<conf> > " + str(stats["min_conf"]) + "` and a"
        "regex signal are kept; otherwise a regex match is kept; FTAN borderline"
        "(0.6 < conf ≤ 0.8 + high moderation confidence) overrides to `H`; as a last"
        "resort a row falls back to `H` only when `origin_label` is not `ftan_<conf>`"
        "and the top `moderation_results` confidence is < 0.7.",
        "",
        "Detection uses obfuscation-tolerant patterns (separators, repeated letters,"
        "homoglyphs, fullwidth, leet, censored-consonants) and a dual-path approach"
        "(original + normalized text). An embedded-slur filter excludes legitimate"
        "words like `niggardly`.",
        "",
        "## Schema",
        "",
        "| column | type | meaning |",
        "|---|---|---|",
        "| `text` | string | input sentence |",
        "| `labels` | list<string> | multi-label subset of {H, H2, HR} |",
        "| `source` | string | originating dataset |",
        "| `origin_label` | string | FTAN offensive label (`ftan_<conf>`) |",
        "| `split_origin` | string | original split of the source |",
        "| `mutated` | int8 | 1 = obfuscation-engine variant |",
        "| `variant` | int8 | variant index within the source row |",
        "| `cluster` | int64 | near-duplicate cluster id |",
        "",
        "## Splits",
        "",
    ]
    for k, v in stats["splits"].items():
        lines.append(f"- `{k}`: {v:,} rows")
    lines += [
        "",
        "## Disclaimer",
        "",
        "This dataset contains raw offensive language. It is intended for research and",
        "moderation-model training only. Labels are regex-derived heuristics, not",
        "human-verified annotations.",
    ]
    (out_dir / "dataset_card.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

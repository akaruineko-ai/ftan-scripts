"""Step 3: Apply the obfuscation mutation engine.

- Offensive rows: keep the original + generate `offensive_variants` mutated
  copies (leet / separators / censoring / homoglyphs / ...) of curse words.
- Clean rows: keep the original; mutate innocuous words (e.g. example -> ex4mp1e)
  in a fraction of rows so orthographic obfuscation is *not* a cue for offense.
- Rows whose offensive text contains no seed word get a sentence-level style
  mutation instead (fullwidth / case shuffle).

Outputs: data/processed/mutated.parquet
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import zlib
from tqdm import tqdm

from common import PROCESSED_DIR, Mutator, clean_text, load_seeds

pd.set_option("future.no_silent_downcasting", True)


def subsample(df: pd.DataFrame, pos_cap: int, neg_cap: int, rng: np.random.Generator):
    if pos_cap <= 0 and neg_cap <= 0:
        return df
    out = []
    for lab, cap in ((1, pos_cap), (0, neg_cap)):
        sub = df[df["label"] == lab]
        if cap > 0 and len(sub) > cap:
            sub = sub.iloc[rng.permutation(len(sub))[:cap]]
        out.append(sub)
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offensive_variants", type=int, default=3)
    ap.add_argument("--clean_variant_frac", type=float, default=0.4)
    ap.add_argument("--max_pos_per_source", type=int, default=0)
    ap.add_argument("--max_neg_per_source", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    seeds, clean_targets = load_seeds()
    mut = Mutator(seed=args.seed)

    src = PROCESSED_DIR / "normalized.parquet"
    df = pd.read_parquet(src)
    print(f"input rows: {len(df):,}")

    if args.max_pos_per_source > 0 or args.max_neg_per_source > 0:
        df = df.groupby("source", group_keys=False).apply(
            lambda g: subsample(g, args.max_pos_per_source, args.max_neg_per_source, rng)
        ).reset_index(drop=True)
        print(f"after per-source cap: {len(df):,}")

    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]

    rows = []
    base_cols = ["text", "label", "source", "origin_label", "split_origin"]

    # ---- offensive: original + mutated variants ----
    print(f"mutating offensive rows ({len(pos):,}) ...")
    for r in tqdm(pos.to_dict("records"), desc="offensive"):
        rows.append({**{c: r[c] for c in base_cols}, "mutated": 0, "variant": 0})
        for v in range(1, args.offensive_variants + 1):
            mut._reseed(args.seed + v * 100_000 + zlib.crc32(f"{r['source']}|{r['text']}".encode()) % 97)
            new_text, hits = mut.mutate_text(r["text"], seeds, max_hits=2, min_hits=1)
            if hits == 0:
                new_text, _ = mut.mutate_random_word(r["text"])
                if new_text == r["text"] and v % 3 == 0:
                    new_text = mut.mutate_sentence_style(r["text"])
            rows.append({
                **{c: r[c] for c in base_cols},
                "text": clean_text(new_text),
                "mutated": 1,
                "variant": v,
            })

    # ---- clean: original + mutated variants (targeting innocuous words) ----
    n_neg_mut = int(len(neg) * args.clean_variant_frac)
    neg_ix = rng.choice(len(neg), size=n_neg_mut, replace=False)
    neg_ix_set = set(neg_ix.tolist())
    print(f"mutating clean rows ({len(neg):,}, frac={args.clean_variant_frac:.2f}) ...")
    for i, r in enumerate(tqdm(neg.to_dict("records"), desc="clean")):
        rows.append({**{c: r[c] for c in base_cols}, "mutated": 0, "variant": 0})
        if i in neg_ix_set:
            mut._reseed(args.seed + 1_000_000 + i)
            new_text, hits = mut.mutate_text(r["text"], clean_targets, max_hits=1, min_hits=1)
            if hits:
                rows.append({
                    **{c: r[c] for c in base_cols},
                    "text": clean_text(new_text),
                    "mutated": 1,
                    "variant": 1,
                })

    out = pd.DataFrame(rows, columns=base_cols + ["mutated", "variant"])
    out = out.drop_duplicates(subset=["text"]).reset_index(drop=True)
    out.to_parquet(PROCESSED_DIR / "mutated.parquet", index=False)

    print(f"\nfinal rows: {len(out):,}")
    print(out["label"].value_counts().sort_index().rename("count").to_string())
    print("\nmutated distribution:")
    print(out.groupby(["label", "mutated"]).size().unstack(fill_value=0).to_string())
    print(f"\nsaved: {PROCESSED_DIR / 'mutated.parquet'}")


if __name__ == "__main__":
    main()

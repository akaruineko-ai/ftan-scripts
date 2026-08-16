"""Shared helpers for the ftan-2.0 data pipeline.

Includes text cleaning, seed-word loading, and the obfuscation mutation
engine used to generate leet/censored/homoglyph variants of offensive terms
(as well as innocent orthographic variation in clean text).
"""

from __future__ import annotations

import json
import random
import re

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FINAL_DIR = DATA_DIR / "final"
SEEDS_DIR = DATA_DIR / "seeds"

UNIFIED_COLUMNS = ["text", "label", "source", "origin_label", "split_origin"]


# --------------------------------------------------------------------------- #
# Text cleaning
# --------------------------------------------------------------------------- #
_WS_RE = re.compile(r"\s+")

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\x00", "")
    return _WS_RE.sub(" ", s).strip()


# --------------------------------------------------------------------------- #
# Seed word loading
# --------------------------------------------------------------------------- #
def load_seeds() -> tuple[set[str], set[str]]:
    """Return (offensive_seeds, clean_targets) as lowercase sets."""
    seeds: set[str] = set()
    for fname in ("ldnoobw_en.txt", "extra_en.txt"):
        p = SEEDS_DIR / fname
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                w = line.strip().lower()
                if len(w) >= 3 and not w.startswith("#"):
                    seeds.add(w)
    clean_targets = set(
        w.lower().strip()
        for w in (
            "example because please before house music people think thanks great hello where "
            "never friend world right would should these those there their thing things want "
            "going watch game play like love life time good back some more very even still "
            "might could about after again against always around other every first found give "
            "know leave make means most need only open over read same seems show small start "
            "state story taken though three through under until water while without word work "
            "years young reason special strong school friends happy money number nothing "
            "morning looking power between family history really better"
        ).split()
    )
    return seeds, clean_targets


_COMMON_SUFFIXES = sorted(
    ["ings", "ing", "ers", "ed", "er", "es", "ies", "s", "y", "ist", "o"],
    key=len,
    reverse=True,
)


def stems_of_word(word: str, targets: set[str]) -> list[str]:
    """Return matching seeds for a word, handling common inflectional suffixes."""
    w = word.lower()
    if w in targets:
        return [w]
    hits = []
    for sfx in _COMMON_SUFFIXES:
        if len(w) > len(sfx) + 2 and w.endswith(sfx) and w[:-len(sfx)] in targets:
            hits.append(w[:-len(sfx)])
            break
    return hits


# --------------------------------------------------------------------------- #
# Mutation engine
# --------------------------------------------------------------------------- #
_LEET = {
    "a": "4", "b": "8", "e": "3", "g": "9", "i": "1", "l": "1",
    "o": "0", "s": "5", "t": "7", "z": "2",
}
_HOMOGLYPH = {
    "a": "\u0430", "e": "\u0435", "o": "\u043e", "c": "\u0441", "p": "\u0440",
    "y": "\u0443", "x": "\u0445", "k": "\u043a", "m": "\u043c", "t": "\u0442",
    "h": "\u043d", "i": "\u0456", "b": "\u044c",
}
_SEPARATORS = [".", "-", "_", "*", " ", "~", "|", "+", "!"]
_VOWELS = set("aeiouy")


def _fullwidth(s: str) -> str:
    out = []
    for ch in s:
        o = ord(ch)
        out.append(chr(o + 0xFEE0) if 0x21 <= o <= 0x7E else ch)
    return "".join(out)


class Mutator:
    """Deterministic, seedable obfuscation engine."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def _reseed(self, seed: int):
        self.rng = random.Random(seed)

    # -- word-level --------------------------------------------------------- #
    def mutate_word(self, word: str) -> str:
        """Apply one random obfuscation style to a single word."""
        style = self.rng.choice(
            ["leet", "sep", "censor", "repeat", "case", "homoglyph", "fullwidth", "combo"]
        )
        return getattr(self, f"_m_{style}")(word.lower())

    def _m_leet(self, w: str) -> str:
        pool = [c for c in w if c in _LEET]
        if not pool:
            return self._m_case(w)
        k = max(1, round(len(pool) * self.rng.uniform(0.3, 0.8)))
        chosen = set(self.rng.sample(pool, k))
        return "".join(_LEET[c] if c in chosen else c for c in w)

    def _m_sep(self, w: str) -> str:
        if len(w) < 4:
            return self._m_case(w)
        sep = self.rng.choice(_SEPARATORS)
        return sep.join(w)

    def _m_censor(self, w: str) -> str:
        # keep first letter (and last if long), star the rest / vowels
        if len(w) <= 2:
            return self._m_case(w)
        mode = self.rng.choice(["middle", "vowels", "spl"])
        if mode == "middle":
            inner = "*" * (len(w) - 1)
            return w[0] + inner
        if mode == "vowels":
            return w[0] + "".join("*" if c in _VOWELS else c for c in w[1:])
        # replace some letters with $ / @ / #
        sub = self.rng.choice("$@#")
        return "".join(sub if c in _VOWELS else c for c in w)

    def _m_repeat(self, w: str) -> str:
        if len(w) < 3:
            return self._m_case(w)
        i = self.rng.randrange(1, len(w))
        times = self.rng.randint(2, 4)
        return w[:i] + w[i] * times + w[i + 1:]

    def _m_case(self, w: str) -> str:
        return "".join(c.upper() if self.rng.random() < 0.5 else c for c in w)

    def _m_homoglyph(self, w: str) -> str:
        pool = [c for c in w if c in _HOMOGLYPH]
        if not pool:
            return self._m_case(w)
        k = max(1, round(len(pool) * self.rng.uniform(0.2, 0.5)))
        chosen = set(self.rng.sample(pool, k))
        return "".join(_HOMOGLYPH[c] if c in chosen else c for c in w)

    def _m_fullwidth(self, w: str) -> str:
        return _fullwidth(w)

    def _m_combo(self, w: str) -> str:
        # stack two cheap transforms deterministically
        r1 = self.rng.random()
        if r1 < 0.5:
            w = self._m_leet(w)
            return self._m_censor(w)
        w = self._m_sep(w)
        return self._m_homoglyph(w)

    # -- sentence-level ----------------------------------------------------- #
    def mutate_sentence_style(self, text: str) -> str:
        """Orthographic-only sentence mutation (no token replacement)."""
        if self.rng.random() < 0.5:
            return _fullwidth(text)
        return "".join(
            c.upper() if c.isalpha() and self.rng.random() < 0.5 else c for c in text
        )

    def mutate_random_word(self, text: str) -> tuple[str, bool]:
        """Apply a random mutation to one random word of the sentence.

        Used as a fallback when no target word is found, so variants stay
        diverse instead of producing identical copies.
        """
        parts = re.split(r"(\s+)", text)
        cand = [
            (i, pre, core, post)
            for i, tok in enumerate(parts)
            for m in [re.match(r"^([^A-Za-z0-9]*)([A-Za-z0-9][A-Za-z0-9]*)([^A-Za-z0-9]*)$", tok)]
            if m and len(m.group(2)) >= 4
            for pre, core, post in [(m.group(1), m.group(2), m.group(3))]
        ]
        if not cand:
            return text, False
        i, pre, core, post = self.rng.choice(cand)
        parts[i] = pre + self.mutate_word(core) + post
        return "".join(parts), True

    def mutate_text(
        self,
        text: str,
        targets: set[str],
        max_hits: int = 2,
        min_hits: int = 1,
    ) -> tuple[str, int]:
        """Replace up to `max_hits` target words with mutated forms.

        Returns (new_text, hit_count). When `min_hits` is not reached and
        fallback=True semantics are wanted, callers decide what to do.
        """
        parts = re.split(r"(\s+)", text)
        hits = 0
        for i, tok in enumerate(parts):
            if hits >= max_hits:
                break
            m = re.match(r"^([^A-Za-z0-9]*)([A-Za-z0-9][A-Za-z0-9]*)([^A-Za-z0-9]*)$", tok)
            if not m:
                continue
            pre, core, post = m.groups()
            stems = stems_of_word(core, targets)
            if not stems:
                continue
            stem = stems[0]
            suffix = core[len(stem):]
            parts[i] = pre + self.mutate_word(stem) + suffix + post
            hits += 1
        return "".join(parts), hits

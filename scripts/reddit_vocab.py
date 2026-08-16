"""Regex vocabulary + sentence classifier for the Reddit expansion pipeline.

The classifier is deliberately narrow: it only detects *explicit* English
profanity and direct insults (the ftan-2.0 task), NOT passive aggression or
sarcasm. Every comment is assigned one of four buckets:

    attack    - profanity / insult aimed at a person (2nd-person pronoun in a
                small window, or an explicit attack pattern)          -> label 1
    emotional - *expletive* profanity with no addressee ("fuck this weather",
                "this is fucking awesome")                             -> label 0
    grey      - a person-directed insult with no 2nd-person pronoun
                ("what an idiot", "he's a moron") -> left to the FTAN
                classifier (curate step)                               -> ??
    clean     - no profanity, no insult                                -> label 0

Word lists are split by *how the word is used*:
  * expletives (fuck, shit, damn, hell, ...) are almost always emotional
    outbursts, so alone they make a comment "emotional";
  * person insults (bitch, asshole, idiot, slurs, ...) are directed at people,
    so without an explicit addressee they land in the grey zone for FTAN.

The profanity patterns tolerate light obfuscation (f*ck, f4ck, f u c k,
f**k, sh*t) so the fetch stage still catches censored curse words; heavier
mutations (homoglyphs, leet, fullwidth) are handled by the FTAN model.
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------- #
# Word lists
# --------------------------------------------------------------------------- #
# Expletives: emotional outbursts, not aimed at anyone.
_EXCLETIVE_WORDS = [
    "fuck", "fucking", "fuckin", "fucked", "fucks",
    "motherfucking",
    "shit", "shitty", "shitting", "shits", "shite", "bullshit", "horseshit",
    "damn", "damned", "dammit", "goddamn", "god damn",
    "hell", "hellish", "bloody",
    "piss", "pissed", "pissing", "pissedoff",
    "wank", "wanking", "wanked", "wanks", "bollocks", "arse",
]

# Person insults: directed at people, even without an explicit "you".
_PERSON_INSULT_WORDS = [
    "bitch", "bitches", "bitchy", "bitching", "bitchass", "sonofabitch",
    "cunt", "cunts",
    "asshole", "assholes", "asshat", "asswipe", "assface", "assclown",
    "assmunch", "asslicker", "arsehole", "arseholes",
    "dick", "dicks", "dickhead", "dickheads", "dickwad", "dickface",
    "cock", "cocks", "cocksucker", "cocksuckers",
    "bastard", "bastards",
    "pussy", "pussies",
    "twat", "twats", "wanker", "wankers",
    "whore", "whores", "slut", "sluts", "skank", "skanks", "hoe", "hoes",
    "thot", "thots", "douchebag", "douchebags", "douche", "douches",
    "jackass", "jackasses",
    "retard", "retards", "retarded",
    "fucker", "fuckers", "motherfucker", "motherfuckers",
    "dipshit", "dumbshit", "shithead", "shitheads", "shitface",
]

_SLURS = [
    "nigger", "niggers", "nigga", "niggas", "niggaz",
    "faggot", "faggots", "fag", "fags",
    "kike", "kikes", "beaner", "beaners", "chink", "chinks", "spic", "spics",
    "wetback", "wetbacks", "gook", "gooks", "dyke", "dykes", "tranny",
    "trannies", "coon", "coons", "spook", "spooks", "cracker", "crackers",
    "towelhead", "raghead", "paki", "jigaboo", "slanteye",
]

_INSULT_WORDS = [
    "idiot", "idiots", "moron", "morons", "imbecile", "imbeciles",
    "dumbass", "dumbasses", "simpleton", "simpletons", "buffoon", "buffoons",
    "numbskull", "dimwit", "dimwits", "nitwit", "nitwits",
    "fool", "fools", "loser", "losers", "clown", "clowns", "scumbag",
    "scumbags", "degenerate", "degenerates", "creep", "creeps",
    "pervert", "perverts", "psycho", "psychos", "cretin", "cretins",
    "stupid", "dumb", "brainless", "braindead", "brain-dead", "worthless",
    "pathetic", "useless", "trashy", "trash", "garbage", "moronic",
    "imbecilic",
]

_ATTACK_PATTERNS = [
    r"\bfuck\s+(?:you|off|u|y\b)\b",
    r"\b(?:go|going|gonna)\s+(?:fuck|screw)\s+(?:yourself|urself)\b",
    r"\bfuck\s+(?:your|ur)\b",
    r"\b(?:screw|kiss)\s+you\b",
    r"\bsuck\s+(?:my|a|your)\b",
    r"\beat\s+shit\b",
    r"\b(?:kys|stfu|gtfo|kms|fuckoff|fuckyou)\b",
    r"\bkill\s+(?:yourself|urself|your\s+self)\b",
    r"\bshut\s+(?:the\s+fuck\s+)?up\b",
    r"\bgo\s+die\b",
    r"\bdie\s+in\s+a\s+fire\b",
    r"\bblow\s+me\b",
    r"\bbite\s+me\b",
    r"\bpiss\s+off\b",
    r"\bfuck\s+this\s+shit\b",
    r"\bpiece\s+of\s+shit\b",
    r"\byou\s+(?:are|re|r)\s+(?:a\s+|an\s+|such\s+a\s+|such\s+an\s+|so\s+|so\s+fucking\s+)?(?:fucking\s+)?(?:idiot|moron|dumbass|retard|retarded|worthless|useless|pathetic|trash|garbage|piece\s+of\s+shit)\b",
    r"\bu\s+(?:r|are)\s+(?:a\s+|an\s+|so\s+)?(?:fucking\s+)?(?:idiot|moron|dumbass|retard|retarded)\b",
]

_SECOND_PERSON_WORDS = [
    "you", "your", "yours", "yourself", "yourselves", "you're", "youre",
    "youll", "youve", "youd", "u", "ur", "ya", "y'all", "yall", "u're",
]


# --------------------------------------------------------------------------- #
# Compiled regexes
# --------------------------------------------------------------------------- #
def _strict_pattern(word: str) -> str:
    """All letters present; only junk (non-word chars / digits / spaces)
    may appear between them. Matches fuck, f u c k, f4ck, fucck."""
    return r"\b" + r"[\W_0-9]*".join(re.escape(ch) for ch in word.lower()) + r"\b"


def _loose_pattern(word: str) -> str:
    """Vowel letters optional, so f*ck / sh*t / b*tch / f**k still match.
    Used as a fallback; the classifier only trusts loose matches when they
    show obfuscation evidence or are long (see _hits)."""
    parts = []
    for ch in word.lower():
        esc = re.escape(ch)
        parts.append(esc + "?" if ch in "aeiouy" else esc)
    return r"\b" + r"[\W_0-9]*".join(parts) + r"\b"


def _build_regexes(words: list[str]) -> tuple[re.Pattern, re.Pattern]:
    return (
        re.compile("|".join(_strict_pattern(w) for w in words), re.IGNORECASE),
        re.compile("|".join(_loose_pattern(w) for w in words), re.IGNORECASE),
    )


EXCLETIVE_STRICT_RE, EXCLETIVE_LOOSE_RE = _build_regexes(_EXCLETIVE_WORDS)
INSULT_STRICT_RE, INSULT_LOOSE_RE = _build_regexes(
    _PERSON_INSULT_WORDS + _SLURS + _INSULT_WORDS
)

ATTACK_RE = re.compile("|".join(_ATTACK_PATTERNS), re.IGNORECASE)
SECOND_PERSON_RE = re.compile(
    r"\b(?:" + "|".join(_SECOND_PERSON_WORDS) + r")\b", re.IGNORECASE
)
# extra catch for heavy vowel+consonant censoring (f**k, sh*t, b**ch, d*mn).
# junk is punctuation/symbols only (no spaces, no digits) and a trailing
# letter group is mandatory, so "a " / "s " do not match.
CENSORED_RE = re.compile(
    "|".join([
        r"\bf[^\w\s]{1,4}k(?:ing|er|ed|s)?\b",
        r"\bs[^\w\s]{1,4}(?:t|tt|ty|ting)\b",
        r"\bb[^\w\s]{1,4}(?:h|ch|tch|chy)\b",
        r"\bc[^\w\s]{1,4}(?:ck|t|nt|n)\b",
        r"\bd[^\w\s]{1,4}(?:ck|k|n|mn)\b",
        r"\ba[^\w\s]{1,4}(?:ss|s)\b",
        r"\bn[^\w\s]{1,4}(?:gg|gga|gger|igga|igg|a)\b",
        r"\bf[^\w\s]{1,4}(?:gg|ggot|ag|ot)\b",
        r"\bh[^\w\s]{1,4}(?:ll|ell)\b",
        r"\bp[^\w\s]{1,4}(?:ssy|iss|ss)\b",
        r"\bw[^\w\s]{1,4}(?:hore|hores)\b",
        r"\bs[^\w\s]{1,4}(?:lut|ut|ank)\b",
    ]),
    re.IGNORECASE,
)
# Unicode word tokens, used for the proximity window below.
WORD_RE = re.compile(r"[^\W_0-9]+")

# max tokens between a 2nd-person pronoun and a profanity/insult word
PROXIMITY_WINDOW = 6
# loose matches this long are trusted even without obfuscation evidence
MIN_LEN_TRUST = 5


def _hits(strict_re, loose_re, text: str) -> list[re.Match]:
    """Matches that genuinely indicate the profanity word.

    A strict match (all letters present) is always trusted. A loose match
    (vowel omitted) is only trusted when it shows obfuscation junk (f*ck) or
    is long enough that the letter skeleton is distinctive.
    """
    out = list(strict_re.finditer(text))
    seen = {m.span() for m in out}
    for m in loose_re.finditer(text):
        if m.span() in seen:
            continue
        s = m.group(0)
        if len(s) >= MIN_LEN_TRUST or re.search(r"[\W_0-9]", s):
            out.append(m)
    return out


def _word_index(words: list[re.Match], char_pos: int) -> int:
    """Index of the word token containing/right after `char_pos`."""
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi) // 2
        if words[mid].end() <= char_pos:
            lo = mid + 1
        else:
            hi = mid
    return lo if lo < len(words) else -1


def classify_regex(text: str) -> dict:
    """Bucket a single comment with pure regex.

    Returns dict with keys: profanity, insult, attack, category
    (one of 'attack', 'emotional', 'grey', 'clean').
    """
    out = {"profanity": False, "insult": False, "attack": False, "category": "clean"}
    if not text:
        return out

    exc_matches = _hits(EXCLETIVE_STRICT_RE, EXCLETIVE_LOOSE_RE, text)
    ins_matches = _hits(INSULT_STRICT_RE, INSULT_LOOSE_RE, text)
    cen_matches = list(CENSORED_RE.finditer(text))

    # censored words count as expletive profanity UNLESS they are also a
    # person insult (b*tch is an insult, f**k is an expletive)
    out["insult"] = bool(ins_matches)
    out["profanity"] = bool(exc_matches) or (bool(cen_matches) and not out["insult"])

    if ATTACK_RE.search(text):
        out["attack"] = True
        out["category"] = "attack"
        return out

    # proximity: profanity/insult within WINDOW tokens of a 2nd-person pronoun
    words = list(WORD_RE.finditer(text))
    second_person_idx = {
        i for i, w in enumerate(words) if SECOND_PERSON_RE.match(w.group(0))
    }
    if second_person_idx:
        hit_idx = {
            i
            for m in exc_matches + ins_matches + cen_matches
            for i in [_word_index(words, m.start())]
            if i >= 0
        }
        if any(abs(p - s) <= PROXIMITY_WINDOW for p in hit_idx for s in second_person_idx):
            out["attack"] = True
            out["category"] = "attack"
            return out

    if out["profanity"]:
        out["category"] = "emotional"
    elif out["insult"]:
        out["category"] = "grey"
    else:
        out["category"] = "clean"
    return out


def prefilter_keep(text: str, clean_sample_frac: float, rng) -> tuple[dict, bool]:
    """Fast fetch-stage decision: keep candidates, drop pure-clean filler.

    Returns (classify dict, keep_bool). Clean rows are kept with probability
    `clean_sample_frac` so the clean bank doesn't balloon to the whole dump.
    """
    info = classify_regex(text)
    if info["category"] != "clean":
        return info, True
    return info, rng.random() < clean_sample_frac
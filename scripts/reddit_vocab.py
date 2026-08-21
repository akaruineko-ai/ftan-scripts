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
import unicodedata

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

# Group-hostile terms: not explicit slurs but terms that in offensive contexts
# indicate hate toward protected groups.  Used via substring OBF matching so
# compound forms (e.g. "glowniggers", "memeflaggot") are caught.  The ftan>0.8
# gate already limits these to offensive rows, keeping FPR low.
_GROUP_HOSTILE = [
    # religion
    "jews", "jewish", "zionist", "zionists", "zionism",
    "muslim", "moslem", "muzzie", "muzzlim",
    "christian", "christians", "catholic", "catholics",
    "evangelical", "fundie", "fundies",
    "atheist", "atheists",
    "gypsy", "gypsies", "romani",
    "infidel", "infidels",
    "kafir", "kafirs", "kuffar",
    "heathen", "heathens",
    "heretic", "heretics",
    "apostate", "apostates",
    # race / ethnicity
    "negro", "negroes", "negroid",
    "mongoloid",
    "asiatic",
    "mexican", "mexicans",
    "illegals", "invaders",
    "subhuman", "sub-humans",
    "savages", "savage",
    "barbarian", "barbarians",
    # dehumanization
    "vermin", "parasite", "parasites",
    "cockroach", "cockroaches",
    "filth", "scum",
    # LGBTQ+
    "homo", "homos",
    "pedophile", "pedophiles", "pedo", "pedos",
    # slurs-as-insults (commonly used but not in _SLURS)
    "retard", "retards", "retarded",
    # political / ideological hate
    "nazi", "nazis", "neo-nazi", "neo-nazis",
    "fascist", "fascists",
    "terrorist", "terrorists",
    "extremist", "extremists",
    "jihadist", "jihadists", "jihad",
    "klansman", "klansmen", "klan",
    "degenerate", "degenerates",
    "libtard", "libtards",
    "cuck", "cucks", "cuckold",
    "soyboy", "soy-boy",
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
    may appear between them. Each letter may repeat (obfuscation `repeat`
    style, e.g. `biiitch`). Matches fuck, f u c k, f4ck, fucck, biiitch."""
    units = [re.escape(ch) + "+" for ch in word.lower()]
    return r"\b" + r"[\W_0-9]*".join(units) + r"\b"


def _loose_pattern(word: str) -> str:
    """Vowel letters optional, so f*ck / sh*t / b*tch / f**k still match.
    Used as a fallback; the classifier only trusts loose matches when they
    show obfuscation evidence or are long (see _hits)."""
    parts = []
    for ch in word.lower():
        esc = re.escape(ch)
        parts.append(esc + "+?" if ch in "aeiouy" else esc + "+")
    return r"\b" + r"[\W_0-9]*".join(parts) + r"\b"


def normalize(text: str) -> str:
    """Reverse the obfuscation engine (scripts/common.py `Mutator`) so regexes
    can match mutated text.

    Order matters:
      1. NFKC normalization maps fullwidth forms (ｆｕｃｋ → fuck) to ASCII.
      2. Translate Cyrillic homoglyphs back to ASCII (а→a, к→k, ...).
      3. Translate leet digits/symbols back to ASCII (4→a, 3→e, 0→o, ...).
      4. Lowercase (regexes are already IGNORECASE, this is just tidiness).

    Separators, repeated letters and remaining digits are intentionally left in
    place — the `_obf_pattern` / `_strict_pattern` matchers tolerate them.
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    from common import _HOMOGLYPH, _LEET  # lazy: common does not import this module
    # reverse the mutation maps: cyrillic/homoglyph value -> ascii key, digit -> ascii key
    t = t.translate({ord(v): k for k, v in _HOMOGLYPH.items()})
    t = t.translate({ord(v): k for k, v in _LEET.items()})
    return t.lower()


def _obf_pattern(word: str, word_boundaries: bool = False) -> re.Pattern:
    """Obfuscation-tolerant, separator-tolerant matcher.

    Each letter may repeat and any non-word/non-digit junk may appear between
    letters, so it catches mutated forms (k_i_l_l, biiitch, ｋｉｌｌ, n39r@).
    Designed to run on `normalize()`d text where fullwidth / homoglyph / leet
    mutations have already been reversed.

    `word_boundaries=True` adds `\\b` on both sides — use for short/ambiguous
    terms (threat words like `gas`, `die`) to avoid false hits inside real
    words (`gasoline`, `diet`). Obfuscated forms are standalone tokens, so they
    still match. `word_boundaries=False` (default) is a substring match, used
    for slurs where embedded forms (`memeflaggot`, `glowniggers`) must be caught.
    """
    units = [re.escape(ch) + "+" for ch in word.lower()]
    body = r"[\W_0-9]*".join(units)
    return re.compile((r"\b" + body + r"\b") if word_boundaries else body, re.IGNORECASE)


# Serious-harm / threat terms (used by the H2 hate/threatening label).
_THREAT_TERMS = [
    "kill", "kys", "kms", "die", "dead", "murder", "murdered", "murdering",
    "rape", "raped", "raping", "shoot", "shot", "stab", "stabbed", "stabbing",
    "beat", "beaten", "beating", "assault", "assaulted", "bomb", "bombed",
    "torture", "tortured", "torturing", "hang", "hanged", "lynch", "lynched",
    "lynching", "gas", "gassed", "exterminate", "exterminated", "genocide",
    "slaughter", "slaughtered", "burn", "burned", "burning", "nuke", "nuked",
    "eradicate", "eradicated", "behead", "beheaded", "crucify", "crucified",
    "slit", "punch", "punched", "decapitate", "decapitated", "strangle",
    "strangled", "bludgeon", "bludgeoned", "massacre", "massacred", "slay",
    "slain",
]
SLUR_OBF_RES = [_obf_pattern(w) for w in _SLURS + _GROUP_HOSTILE]
THREAT_OBF_RES = [_obf_pattern(w, word_boundaries=True) for w in _THREAT_TERMS]


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
# Generic censored-word collapse: strips runs of *, $, @, # between letters
# so "n****r", "f**k", "b*tch" become "nr", "fk", "btch".  A follow-up
# first/last-letter check against known slur skeletons handles these.
GENERIC_CENSORED_RE = re.compile(r"[\*\$@#]{1,8}")

# Known slur skeletons: (first_char, last_char, min_remaining_length).
# After stripping censored chars, the first/last letters must match and the
# remaining string must be long enough.  This catches "n****r" (n...r),
# "f*ck" (f...k), "b*tch" (b...h) etc.
_SLUR_SKELETONS = [
    ("n", "r", 4),   # nigger, negro
    ("f", "k", 3),   # fuck
    ("f", "t", 5),   # faggot
    ("b", "h", 4),   # bitch
    ("s", "t", 4),   # shit
    ("c", "k", 3),   # cock
    ("d", "k", 3),   # dick
    ("p", "y", 4),   # pussy
    ("w", "e", 5),   # whore
    ("s", "k", 4),   # skank
    ("r", "d", 5),   # retard
    ("k", "e", 3),   # kike
    ("c", "k", 3),   # chink (ch...k)
]


def _de_censored(text: str) -> str:
    """Strip runs of *, $, @, # from the text so censored-consonant forms
    like 'n****r', 'f**k', 'b*tch' become 'nr', 'fk', 'btch' which the obf
    patterns can match via separator-tolerant matching.
    """
    return GENERIC_CENSORED_RE.sub("", text)


def _build_censored_patterns() -> list[re.Pattern]:
    """Build regex patterns that match censored slurs via first/last letter.

    For each word in the offensive lexicons (slurs, group-hostile, insults,
    expletives), build: first_char + [*$@#]{1,12} + last_char.
    This matches 'n****r', 'f*ck', 'b*tch' etc.  Uses word boundaries to
    avoid false positives.
    """
    all_words = set(_SLURS + _GROUP_HOSTILE + _PERSON_INSULT_WORDS + _INSULT_WORDS
                    + _EXCLETIVE_WORDS)
    patterns = []
    seen = set()
    for w in all_words:
        if len(w) < 3:
            continue
        key = (w[0], w[-1])
        if key in seen:
            continue
        seen.add(key)
        pat = re.compile(
            r"\b" + re.escape(w[0]) + r"[\*\$@#]{1,12}" + re.escape(w[-1]) + r"\b",
            re.IGNORECASE,
        )
        patterns.append(pat)
    return patterns


_CENSORED_SLUR_RES = _build_censored_patterns()


def _matches_censored_slur(text: str) -> bool:
    """Check if text contains a censored slur via first/last letter matching.

    Matches patterns like 'n****r' (n + 4 stars + r), 'f*ck' (f + star + ck),
    'b*tch' (b + star + tch).  Uses first/last letter from known slurs with
    word boundaries to avoid false positives.
    """
    return any(p.search(text) for p in _CENSORED_SLUR_RES)


# Words that legitimately contain a slur as a substring (e.g. niggardly,
# triggered).  These are excluded when the match is a *substring* of a longer
# common word.  Checked via simple suffix/prefix heuristics rather than a
# full dictionary lookup to keep the filter lightweight.
_LEGITIMATE_SUFFIXES = (
    "ly", "ness", "hood", "ism", "ist", "able", "ible",
    "ing", "tion", "ment", "ance", "ence",
)
_LEGITIMATE_PREFIXES = (
    "tra", "un", "re", "dis", "over", "under", "mis",
)


def _is_embedded_slur(match_start: int, match_end: int, text: str) -> bool:
    """Return True if the match is embedded inside a longer common word.

    'nigger' inside 'niggardly' -> True (should be excluded).
    'nigger' standalone or at word boundary -> False (keep the match).
    """
    if match_start > 0 and text[match_start - 1].isalpha():
        # preceded by a letter: extract the contiguous alpha run before the match
        before = ""
        for ch in text[max(0, match_start - 12):match_start]:
            if ch.isalpha():
                before += ch
            else:
                before = ""  # reset on non-alpha
        if any(before.endswith(p) for p in _LEGITIMATE_PREFIXES):
            return True
    if match_end < len(text) and text[match_end].isalpha():
        # followed by a letter: extract the contiguous alpha run after the match
        after = ""
        for ch in text[match_end:match_end + 12]:
            if ch.isalpha():
                after += ch
            else:
                break
        if any(after.endswith(s) for s in _LEGITIMATE_SUFFIXES):
            return True
    return False


def normalize_full(text: str) -> str:
    """Combined normalization: NFKC + homoglyph + leet + lowercase + de-censor.

    This is the primary entry point for converting raw (possibly obfuscated)
    text into a form that the OBF patterns can match.
    """
    t = normalize(text)
    t = _de_censored(t)
    return t
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
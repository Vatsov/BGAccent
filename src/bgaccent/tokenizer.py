from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

TokenKind = Literal["word", "space", "punct", "number", "abbrev", "other"]


@dataclass(frozen=True, slots=True)
class Token:
    kind: TokenKind
    text: str
    original: str
    line: int
    col: int


_ABBREVS = {
    "д-р",
    "проф.",
    "т.е.",
    "т.н.",
    "напр.",
    "ж.п.",
    "бул.",
    "акад.",
    "доц.",
    "инж.",
    "гр.",
    "ул.",
    "ж.к.",
}

_ABBREV_PATTERN = "|".join(re.escape(a) for a in sorted(_ABBREVS, key=len, reverse=True))

_TOKEN_RE = re.compile(
    r"(?P<abbrev>" + _ABBREV_PATTERN + r")"
    r"|(?P<word>(?=\S*[a-zA-ZЀ-ӿ])(?:[\wЀ-ӿ]+-)*[\wЀ-ӿ]+)"
    r"|(?P<number>\d+)"
    r"|(?P<space>\s+)"
    r"|(?P<punct>[^\w\sЀ-ӿ])"
    r"|(?P<other>.)",
    re.UNICODE,
)


def tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    line = 1
    col = 1

    for match in _TOKEN_RE.finditer(text):
        matched_text = match.group()
        kind = match.lastgroup
        assert kind is not None

        tokens.append(Token(
            kind=kind,  # type: ignore[arg-type]
            text=matched_text,
            original=matched_text,
            line=line,
            col=col,
        ))

        for ch in matched_text:
            if ch == "\n":
                line += 1
                col = 1
            else:
                col += 1

    return tokens


def detokenize(tokens: list[Token]) -> str:
    return "".join(t.original for t in tokens)

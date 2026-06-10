"""Context-aware forbidden-claim scanner for ECHO-Bench (Task G-005).

This module is the *automated* form of the ``validate-claims`` guardrail check.
It scans English claim text in ``docs/`` and ``outputs/reports/`` for the
forbidden-phrase set defined in this module (the project guardrail tokens) and
reports each genuine, **assertion-style** violation with its file, line number,
matched phrase, and the line text.

Why context-awareness (the heuristic)
-------------------------------------
A naive substring/grep scan over the same phrase list produces FALSE POSITIVES
on two legitimate constructs that the project relies on:

1. **Disclaimers / negations.** The guardrail docs deliberately *deny* the
   forbidden claims, e.g. "this is NOT a privacy guarantee", "not an emotion or
   wellbeing judgment", "the system forbids any persona / emotion / preference
   field". A naive scan flags the very text that enforces the guardrail.
2. **Identifiers / config keys / policy names.** The codebase contains the policy
   name ``PSEUDO_USER_MODEL``, the trace key ``user_model``, the forbidden-field
   name ``preference_vector``, the metric ``salience_outlier_rate``, etc. These
   are machine-read identifiers, not prose claims.

So a match is reported ONLY when the phrase is used as an **assertion** and is
NOT, additionally:

- preceded (on the same line, before the match) by a **negator** such as
  ``not`` / ``no`` / ``never`` / ``without`` / ``excludes`` / ``forbids`` /
  ``rejects`` / ``rather than`` / ``not a`` / "carries no", etc.;
- part of an **enumerated forbidden-claims / forbidden-fields list** — either a
  line inside the ``## Forbidden Claims`` section of ``docs/01_GUARDRAILS.md`` or
  a slash/comma-separated forbidden-field enumeration introduced by a denial word
  (``forbidden`` / ``forbids`` / ``no`` / ``not`` / ``without`` / ``reject`` /
  ``rule out`` ...);
- part of a **code identifier / config key / policy name** — the matched phrase
  sits inside an underscore_token, a camelCase token, or a backtick/quote-wrapped
  identifier (e.g. ``user_model``, ``PSEUDO_USER_MODEL``, ``preference_vector``,
  ``salience_outlier_rate``).

This is a **guardrail aid, not a guarantee**: it is a deterministic heuristic
that catches the common claim-style violations and suppresses the known
legitimate constructs. It does not replace human review of novel phrasings.

CLI
---
``python -m echo_bench.tools.claim_check [paths...]`` scans the default targets
(``docs/`` and ``outputs/reports/``) when no paths are given, prints any genuine
findings, emits a Korean summary log line, and exits non-zero if any genuine
assertion-style forbidden claim is found (exit ``0`` when clean).

All identifiers and file paths stay English; the summary log line is Korean per
the project logging convention.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

__all__ = [
    "FORBIDDEN_PHRASES",
    "Finding",
    "scan_text",
    "scan_path",
    "scan_paths",
    "main",
]

# ---------------------------------------------------------------------------
# Forbidden-phrase set
# ---------------------------------------------------------------------------
# The project guardrail forbidden-claim tokens (emotion / persona / preference
# used as a CLAIM). Matched case-insensitively as whole words/phrases. Suppression of
# disclaimers, enumerations, and identifiers is handled by the context heuristic
# below, NOT by removing phrases from this list (allowlisting a phrase to pass is
# forbidden by G-005).
FORBIDDEN_PHRASES: tuple[str, ...] = (
    # validate-claims.md
    "users prefer",
    "users understand",
    "emotional",
    "wellbeing",
    "therapeutic",
    "gdpr compliant",
    "personality",
    "diagnosis",
    "meaning-free",
    "interpretation-free",
    # guardrail claim tokens
    "emotion",
    "persona",
    "preference",
)


@dataclass(frozen=True)
class Finding:
    """One genuine assertion-style forbidden-claim hit.

    Attributes:
        file: path to the file the hit was found in (English).
        line: 1-based line number.
        phrase: the forbidden phrase that matched (from FORBIDDEN_PHRASES).
        text: the full text of the offending line (stripped of trailing newline).
    """

    file: str
    line: int
    phrase: str
    text: str


# ---------------------------------------------------------------------------
# Context heuristics
# ---------------------------------------------------------------------------

# Negators that, when they appear before the match on the same line, mark the
# phrase as a denial / disclaimer rather than an assertion.
_NEGATORS: tuple[str, ...] = (
    "not",
    "no",
    "never",
    "without",
    "excludes",
    "exclude",
    "excluding",
    "forbids",
    "forbid",
    "forbidden",
    "rejects",
    "reject",
    "rejecting",
    "refuses",
    "refuse",
    "rules out",
    "rule out",
    "neither",
    "nor",
    "non",
    "nothing",
    "none",
    "free of",
    "absent",
    "prohibits",
    "prohibit",
    "prohibited",
    "denies",
    "deny",
    "disallow",
    "disallows",
    "represents",
    "infers",
    "models",
    "measures of",
    "measure of",
    "claim",
    "claims",
)

# A negation phrase counts if any negator token appears in the text preceding the
# match (same line). We use word-boundary matching so "note" does not match "not".
_NEGATOR_RE = re.compile(
    r"(?<![A-Za-z])(?:"
    + "|".join(re.escape(n) for n in _NEGATORS)
    + r")(?![A-Za-z])",
    re.IGNORECASE,
)

# Words that, when present anywhere on the line, mark a slash/comma-separated
# enumeration of forbidden FIELDS/CLAIMS as a denial list (e.g.
# "forbids any user / persona / emotion / preference field").
_ENUM_DENIAL_WORDS: tuple[str, ...] = (
    "forbidden",
    "forbids",
    "forbid",
    "no ",
    "not ",
    "without",
    "reject",
    "refuse",
    "rule out",
    "rules out",
    "free of",
    "free-text",
    "free text",
    "latent",
    "leak",
    "metadata",
    "field",
    "fields",
    "claim",
    "claims",
    "framing",
    "judgment",
    "guardrail",
    "user model",
    "user-model",
    "represents",
    "infers",
    "models a person",
    "models nothing",
    # scope-restriction / definition cues that govern an enumerated list
    "measures of",
    "measure of",
    "list of",
    "set of",
    "any user",
    "user's",
    "satisfaction",
    "latent representation",
    "intent",
    "token",
    "key",
)

# Heading line that opens the docs/01_GUARDRAILS.md "Forbidden Claims" block.
_FORBIDDEN_SECTION_HEADING_RE = re.compile(
    r"^\s*#+\s*forbidden\b", re.IGNORECASE
)
# Any markdown heading line (closes a section).
_HEADING_RE = re.compile(r"^\s*#+\s+\S")


def _match_iter(text: str, phrase: str) -> Iterable[re.Match]:
    """Yield whole-token matches of ``phrase`` in ``text`` (case-insensitive).

    Boundaries treat ``_`` and alphanumerics as identifier characters so that a
    phrase embedded in an identifier (``user_model``, ``preference_vector``) is
    still matched here — the identifier suppression is applied separately so we
    can distinguish "found inside an identifier" from "found in prose".
    """
    pat = re.compile(
        r"(?<![A-Za-z0-9])" + re.escape(phrase).replace(r"\ ", r"\s+") + r"(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    return pat.finditer(text)


def _is_identifier_context(line: str, start: int, end: int) -> bool:
    r"""True if the match is part of a code identifier / config key / quoted token.

    Detects:

    - an adjacent ``_`` (snake_case token, e.g. ``user_model``,
      ``preference_vector``, ``salience_outlier_rate``);
    - a camelCase / digit join on either side (e.g. ``userModel``);
    - the phrase wrapped in **backticks** — markdown code spans mark machine-read
      identifiers / field names / config keys, never prose claims (e.g.
      ``\`user_model\``, ``\`emotion\``, ``\`preference\``);
    - the phrase wrapped in quotes as a snake_case / camelCase / ALL_CAPS
      identifier (e.g. ``"user_model"``).
    """
    before = line[:start]
    after = line[end:]

    # Adjacent underscore => part of a snake_case identifier.
    if before.endswith("_") or after.startswith("_"):
        return True

    # camelCase / alnum join on either side.
    if before and before[-1].isalnum():
        return True
    if after and after[:1].isalnum():
        return True

    # Backtick code span: a markdown identifier / field name / config key.
    if before.rfind("`") != -1 and after.find("`") != -1:
        return True

    # Quote-wrapped token that itself looks like a code identifier.
    for quote in ('"', "'"):
        lb = before.rfind(quote)
        ra = after.find(quote)
        if lb == -1 or ra == -1:
            continue
        token = line[lb + 1 : end + ra]
        if "_" in token or re.search(r"[a-z][A-Z]", token) or token.isupper():
            return True
    return False


def _is_negated(line: str, start: int) -> bool:
    """True if a negator precedes the match on the same line (disclaimer)."""
    return _NEGATOR_RE.search(line[:start]) is not None


def _line_is_enumeration(line: str, start: int, end: int) -> bool:
    """True if the match sits inside a slash/comma-separated token list."""
    window = line[max(0, start - 40) : min(len(line), end + 40)]
    return bool(re.search(r"[\w-]+\s*[/,]\s*[\w-]+", window)) and (
        "/" in window or "," in window
    )


def _is_denial_enumeration(line: str, start: int, end: int) -> bool:
    """True if the match sits in a slash/comma-separated forbidden-field list.

    Recognises enumerations like
    ``user / persona / emotion / preference / free-text`` or
    ``preference, intent, emotion, persona`` that are introduced (anywhere on the
    line) by a denial / forbidden-field / scope-restriction cue.
    """
    if not _line_is_enumeration(line, start, end):
        return False
    low = line.lower()
    return any(word in low for word in _ENUM_DENIAL_WORDS)


def scan_text(text: str, *, file: str = "<text>") -> List[Finding]:
    """Scan ``text`` and return genuine assertion-style forbidden-claim findings.

    A line is reported for a phrase only when the phrase is used as an ASSERTION:
    it is NOT a negation/disclaimer, NOT part of an enumerated forbidden-claims
    section or denial list, and NOT part of a code identifier / config key /
    policy name. See the module docstring for the full heuristic.

    Args:
        text: the document text to scan.
        file: label recorded on each :class:`Finding` (defaults to ``<text>``).

    Returns:
        A deterministically ordered list of :class:`Finding` (by line, then by
        offset, then by phrase order).
    """
    findings: List[Finding] = []
    lines = text.splitlines()

    # Track whether we are inside the docs/01_GUARDRAILS.md "Forbidden ..."
    # enumeration section (between its heading and the next heading): every
    # bullet there enumerates a forbidden claim, never asserts one.
    in_forbidden_section = False

    # Previous non-blank line + whether it carried a denial / enumeration cue, so
    # a clause that *wraps* onto the next line (e.g. "... may not read or emit a
    # user_id, persona, ...\n diagnosis label.") stays suppressed.
    prev_line = ""
    prev_was_denial_context = False

    for idx, raw_line in enumerate(lines, start=1):
        if _HEADING_RE.match(raw_line):
            in_forbidden_section = bool(
                _FORBIDDEN_SECTION_HEADING_RE.match(raw_line)
            )

        line = raw_line.rstrip("\n")
        if in_forbidden_section:
            # Inside an explicit forbidden-claims enumeration: suppress all.
            prev_line = line
            prev_was_denial_context = True
            continue

        # A line continues a denial/enumeration clause when the previous
        # non-blank line carried such a cue AND its clause was still open (it did
        # not end in a sentence terminator), so a forbidden-field list that wraps
        # across the line break (e.g. "... they are not measures\n of user
        # preference, ... wellbeing, ...") stays suppressed.
        prev_stripped = prev_line.rstrip()
        prev_clause_open = bool(prev_stripped) and not prev_stripped.endswith(
            (".", ":", ";", "!", "?")
        )
        continues_denial = prev_was_denial_context and prev_clause_open
        # The previous (open) line ended on a list connector, so this line is the
        # tail of a wrapped enumeration (e.g. "... free text, or\n diagnosis
        # label.") even when this fragment is not itself comma/slash-separated.
        prev_ends_list_connector = continues_denial and bool(
            re.search(r"(?:,|/|\bor|\band|\bnor)\s*$", prev_stripped, re.IGNORECASE)
        )

        line_low = line.lower()
        line_has_negator = _NEGATOR_RE.search(line) is not None
        line_has_denial_word = any(w in line_low for w in _ENUM_DENIAL_WORDS)

        for phrase in FORBIDDEN_PHRASES:
            for m in _match_iter(line, phrase):
                start, end = m.start(), m.end()
                if _is_identifier_context(line, start, end):
                    continue
                if _is_negated(line, start):
                    continue
                if _is_denial_enumeration(line, start, end):
                    continue
                # Suppress a wrapped enumeration fragment whose denial cue lived
                # on the previous (still-open) line, as long as this fragment
                # introduces no fresh assertion verb of its own.
                if (
                    continues_denial
                    and (
                        _line_is_enumeration(line, start, end)
                        or prev_ends_list_connector
                    )
                    and not re.search(
                        r"\b(is|are|was|were|prefer|improve|improves|show|shows"
                        r"|prove|proves|demonstrate|demonstrates)\b",
                        line_low,
                    )
                ):
                    continue
                findings.append(
                    Finding(file=file, line=idx, phrase=phrase, text=line.strip())
                )

        if line.strip():
            prev_line = line
            prev_was_denial_context = line_has_negator or line_has_denial_word

    findings.sort(key=lambda f: (f.line, f.text, FORBIDDEN_PHRASES.index(f.phrase)))
    return findings


def scan_path(path: str | Path) -> List[Finding]:
    """Scan a single file and return its genuine findings.

    Non-text / unreadable files are skipped (returns an empty list). The file
    path string is recorded on each finding.
    """
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    return scan_text(text, file=str(p))


# File suffixes scanned by default (claim text lives in markdown + JSON reports).
_SCAN_SUFFIXES: tuple[str, ...] = (".md", ".json", ".txt")


def _iter_files(target: Path) -> Iterable[Path]:
    """Yield scannable files under ``target`` (recursively if it is a dir)."""
    if target.is_dir():
        for child in sorted(target.rglob("*")):
            if child.is_file() and child.suffix.lower() in _SCAN_SUFFIXES:
                yield child
    elif target.is_file():
        yield target


def scan_paths(paths: Sequence[str | Path]) -> List[Finding]:
    """Scan multiple files/directories and return all genuine findings.

    Directories are walked recursively for ``.md`` / ``.json`` / ``.txt`` files.
    Findings are returned in a deterministic order (by file, then line).
    """
    findings: List[Finding] = []
    seen: set[Path] = set()
    for raw in paths:
        target = Path(raw)
        for f in _iter_files(target):
            resolved = f.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            findings.extend(scan_path(f))
    findings.sort(key=lambda f: (f.file, f.line, f.text))
    return findings


def _default_targets() -> List[Path]:
    """Return the default scan targets: ``docs/`` and ``outputs/reports/``."""
    repo_root = Path(__file__).resolve().parents[3]
    return [repo_root / "docs", repo_root / "outputs" / "reports"]


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Scan targets, print findings, return an exit code.

    Args:
        argv: optional path arguments; when empty, scans the default targets
            (``docs/`` and ``outputs/reports/``).

    Returns:
        ``0`` when the scanned tree is clean; ``1`` when any genuine
        assertion-style forbidden claim is found.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    targets: Sequence[str | Path] = argv if argv else _default_targets()

    findings = scan_paths(targets)

    if findings:
        for f in findings:
            # Identifiers / paths stay English.
            print(f"{f.file}:{f.line}: forbidden claim '{f.phrase}': {f.text}")
        # Korean summary line.
        print(
            f"클레임 스캔 실패: 금지된 주장 {len(findings)}건 발견 "
            "(어서션형 표현만 보고; 면책/식별자는 제외됨)"
        )
        return 1

    # Korean summary line.
    print("클레임 스캔 통과: 금지된 주장 없음 (0건)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

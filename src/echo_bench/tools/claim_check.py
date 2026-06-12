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

G-008 / TRD G-012 — Oracle-terminology guardrail
-------------------------------------------------
Oracle policies are **objective-specific references**, never global optima. The
phrase set ``"global upper bound"``, ``"global optimum"``, ``"globally optimal"``
is forbidden in assertion-style claim text (docs / reports). Negated forms (e.g.
the exact ``REFERENCE_NOTE`` string ``"objective-specific reference, not global
optimum"``) are suppressed by the existing negator heuristic.

In addition, when scanning a JSON report file the function
:func:`check_oracle_note` enforces a JSON-level rule: if a report carries both
``oraclePolicy`` and ``oraclePolicyDisplayName`` (post-C-014 format), the
``oracleNote`` field must equal ``REFERENCE_NOTE`` exactly. Legacy reports that
have ``oraclePolicy`` but no ``oraclePolicyDisplayName`` emit a Korean warning
log but do NOT fail.

G-010 — Leakage-improvement / privacy claim-FORM patterns
----------------------------------------------------------
Claim ladder v2 (``docs/12_CLAIM_LADDER.md``, ``ladderVersion: claim-ladder-2``)
permanently quarantines leakage-improvement-style and privacy-style claim
FORMS. :data:`FORBIDDEN_CLAIM_PATTERNS` holds case-insensitive **regex**
patterns (vs. the literal phrases in :data:`FORBIDDEN_PHRASES`) that target
these forms in English claim text:

- ``reduce(?:s|d)?\\s+leakage`` / ``improv(?:e|es|ed|ing)\\s+leakage``
  (leakage-improvement claims);
- ``leaks?\\s+user\\s+information`` (user-information-leak claims);
- ``privacy[-\\s]preserving`` (privacy-guarantee claims);
- ``is\\s+privacy\\s+leakage`` (equating probe separability with privacy
  leakage).

Pattern matches run through the SAME context heuristics as the phrase list, so
the established quoting mechanisms stay scanner-safe: a **backtick code span**
(mention, not assertion — the mechanism the ladder's forbidden-forms list uses),
a **negated / denial** line, and the ``## Forbidden ...`` enumeration section
are all suppressed. When a pattern fires, the CLI prints a Korean guidance
message naming the pattern and the required reframing: the statement must be
rewritten as a **probe separability diagnostic** (approved replacement
sentences are printed; see ``docs/12_CLAIM_LADDER.md`` Section 3).

G-022a — Track M mechanism-claim layer
---------------------------------------
Mechanism-CAUSAL sentences about exploration / probe separability are
forbidden UNLESS licensed. :data:`MECHANISM_CLAIM_PATTERNS` holds EXACTLY the
five preregistered claim-form regexes (no broadening). Decision per SENTENCE
containing a mechanism pattern (sentences are reassembled across wrapped
lines; backtick/identifier mentions are suppressed as elsewhere):

1. The sentence carries a HYPOTHESIS marker
   (:data:`MECHANISM_HYPOTHESIS_MARKERS`, case-insensitive) → allowed (M1).
2. Else the sentence carries BOTH SCOPE markers
   (:data:`MECHANISM_SCOPE_MARKERS`) → M2 form: allowed ONLY when an M2
   license is active (resolved by ``echo_bench.tools.ladder_gate.
   evaluate_mechanism_license`` from prereg + reports + ledger evidence;
   NEVER from licenses.json). When the license carries ``caveatRequired``,
   the sentence must ALSO contain the canonical caveat marker from the
   prereg (``tieBreakCaveatMarker``) as an EXACT, case-sensitive substring.
3. Else → finding (fail closed; rewrite as M1 hypothesis or obtain the two
   scope markers + a ladder_gate M2 license).

A sentence containing a mechanism pattern AND ``user``/``users``/``privacy``
additionally emits a Korean WARNING line (stdout only — never a finding,
never affects the exit code).

CLI
---
``python -m echo_bench.tools.claim_check [paths...]`` scans the default targets
(``docs/`` and ``outputs/reports/``) when no paths are given, prints any genuine
findings, emits a Korean summary log line, and exits non-zero if any genuine
assertion-style forbidden claim is found (exit ``0`` when clean).

Optional Track M gate arguments: ``--prereg P --reports R1 R2... --ledger L
[--release]``. When ALL THREE are provided the M2 license is recomputed once
up front via ``ladder_gate.evaluate_mechanism_license``; otherwise (default)
no license is active and M2-form sentences are findings (fail closed).

All identifiers and file paths stay English; the summary log line is Korean per
the project logging convention.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from echo_bench.logging import get_logger
from echo_bench.policies.display_names import REFERENCE_NOTE

__all__ = [
    "FORBIDDEN_PHRASES",
    "FORBIDDEN_CLAIM_PATTERNS",
    "MECHANISM_CLAIM_PATTERNS",
    "MECHANISM_CLAIM_PATTERNS_V3",
    "MECHANISM_HYPOTHESIS_MARKERS",
    "MECHANISM_SCOPE_MARKERS",
    "DEFAULT_TIE_BREAK_CAVEAT_MARKER",
    "OracleNoteViolation",
    "Finding",
    "check_oracle_note",
    "scan_text",
    "scan_path",
    "scan_paths",
    "main",
]

_logger = get_logger(__name__)

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
    # G-008 / TRD G-012: oracle-terminology tokens.
    # Oracle policies are objective-specific references, never global optima.
    # Negated forms (e.g. REFERENCE_NOTE: "...not global optimum") are
    # suppressed by the existing negator heuristic.
    "global upper bound",
    "global optimum",
    "globally optimal",
)

# G-010 (claim ladder v2): forbidden claim-FORM regex patterns. Unlike
# FORBIDDEN_PHRASES (literal phrases), these are case-insensitive regexes that
# target the permanently quarantined leakage-improvement / privacy claim FORMS
# (docs/12_CLAIM_LADDER.md Section 4). They are matched per line with the same
# whole-token boundaries and run through the same suppression heuristics
# (negation, denial enumeration, backtick code spans, Forbidden section), so
# the ladder's own backtick-quoted forbidden-forms list never trips the scan.
FORBIDDEN_CLAIM_PATTERNS: tuple[str, ...] = (
    # Leakage-improvement claims ("X reduces/improves leakage").
    r"reduce(?:s|d)?\s+leakage",
    r"improv(?:e|es|ed|ing)\s+leakage",
    # User-information-leak claims ("X leaks user information").
    r"leaks?\s+user\s+information",
    # Privacy-guarantee claims ("the system is privacy-preserving").
    r"privacy[-\s]preserving",
    # Equating probe separability with privacy leakage.
    r"is\s+privacy\s+leakage",
)

# Compiled forms with the same whole-token boundary guards as _match_iter.
_FORBIDDEN_PATTERN_RES: tuple[tuple[str, "re.Pattern[str]"], ...] = tuple(
    (
        pattern,
        re.compile(
            r"(?<![A-Za-z0-9])(?:" + pattern + r")(?![A-Za-z0-9])",
            re.IGNORECASE,
        ),
    )
    for pattern in FORBIDDEN_CLAIM_PATTERNS
)

# G-022a (Track M): mechanism-causal claim-form regex patterns. EXACTLY the
# five preregistered forms — no broadening, no additions. Matched per SENTENCE
# (reassembled across wrapped lines) with the same whole-token boundary guards
# as FORBIDDEN_CLAIM_PATTERNS; backtick/identifier mentions are suppressed via
# _is_identifier_context. A match is a finding UNLESS the sentence is licensed
# (M1 hypothesis marker, or M2 scope markers + active ladder_gate license).
MECHANISM_CLAIM_PATTERNS: tuple[str, ...] = (
    r"requires\s+history[-\s]dependent\s+exploration",
    r"attributable\s+to\s+(?:the\s+)?(?:history[-\s]dependent\s+)?exploration",
    r"driven\s+by\s+(?:the\s+)?(?:history[-\s]dependent\s+)?exploration",
    r"caus(?:ed|es|ing)\s+[^.\n]{0,80}probe[-\s]separability",
    r"not\s+adaptivity\s+itself",
)

# Compiled forms with the same whole-token boundary guards as
# _FORBIDDEN_PATTERN_RES.
_MECHANISM_PATTERN_RES_V1: tuple[tuple[str, "re.Pattern[str]"], ...] = tuple(
    (
        pattern,
        re.compile(
            r"(?<![A-Za-z0-9])(?:" + pattern + r")(?![A-Za-z0-9])",
            re.IGNORECASE,
        ),
    )
    for pattern in MECHANISM_CLAIM_PATTERNS
)

# G-022a-v3 (N2): Track M v3 "imprint-washout" vocabulary patterns. Added as
# a NEW tuple so that MECHANISM_CLAIM_PATTERNS remains BYTE-IDENTICAL (existing
# tests pin it exactly). All patterns use bounded gaps [^.\n]{0,N} only; causal
# verb + mechanism noun coupling only; plain technical prose must pass.
#
# N2 review fixes applied here:
#   Fix 1 (Gap-spec): frozen\s+.{0,40}state → frozen\s+[^.\n]{0,40}state (both
#     amplify patterns). The comment above now accurately reflects the invariant.
#   Fix 2 (entry-point over-suppression): entry[-\s]point removed from the shared
#     v2-era noun group; replaced by two dedicated patterns (verb-first and
#     noun-first) that require a mechanism-context anchor token
#     (?:separabilit|probe|slate|mechanism) within [^.\n]{0,80} of entry-point.
#     Plain CLI prose ("The CLI entry-point requires three positional arguments.")
#     carries no such anchor and therefore passes.
#   Fix 3a (washout bare coupling): washout split out of the imprint/washout group;
#     the washout-specific patterns require a mechanism-context anchor token
#     (?:separabilit|probe|slate|trace|polic|state|imprint) in the same bounded
#     window. imprint(?:ing)? remains in its own patterns without the anchor
#     requirement (it is inherently mechanism-specific).
#   Fix 3b (noun-first freeze bare coupling): freez(?:e|ing) split out of the
#     noun-first amplify group; the bare-freeze noun-first pattern requires a
#     mechanism-context anchor token (?:trace|polic|state|imprint|separabilit|
#     probe|slate) in [^.\n]{0,160} before freeze OR [^.\n]{0,80} between freeze
#     and amplif OR [^.\n]{0,80} after amplif. Inherently-mechanism nouns
#     (trace[-\s]conditioned, frozen\s+[^.\n]{0,40}state, imprint(?:ing)?) keep
#     their own noun-first pattern without the anchor requirement.
MECHANISM_CLAIM_PATTERNS_V3: tuple[str, ...] = (
    # amplification coupling: amplify verb ↔ trace-conditioned / frozen state / imprint
    # (verb-first) — freez(?:e|ing) retained here; amplif is the anchor itself
    r"amplif(?:y|ies|ied)[^.\n]{0,80}"
    r"(?:trace[-\s]conditioned|frozen\s+[^.\n]{0,40}state|freez(?:e|ing)|imprint(?:ing)?)",
    # (noun-first, inherently mechanism-specific nouns — no bare freeze)
    r"(?:trace[-\s]conditioned|frozen\s+[^.\n]{0,40}state|imprint(?:ing)?)"
    r"[^.\n]{0,80}amplif(?:y|ies|ied)",
    # (noun-first, bare freeze — requires mechanism-context anchor in sentence)
    r"(?:trace|polic|state|imprint|separabilit|probe|slate)[^.\n]{0,160}freez(?:e|ing)[^.\n]{0,80}amplif(?:y|ies|ied)"
    r"|freez(?:e|ing)[^.\n]{0,80}(?:trace|polic|state|imprint|separabilit|probe|slate)[^.\n]{0,80}amplif(?:y|ies|ied)"
    r"|freez(?:e|ing)[^.\n]{0,80}amplif(?:y|ies|ied)[^.\n]{0,80}(?:trace|polic|state|imprint|separabilit|probe|slate)",
    # elimination coupling: eliminate ↔ trace-independent bonus/score/randomization
    # (verb-first)
    r"eliminat(?:e|es|ed|ion)[^.\n]{0,80}"
    r"trace[-\s]independent[^.\n]{0,40}(?:bonus|score|randomi[sz]ation)",
    # (noun-first)
    r"trace[-\s]independent[^.\n]{0,40}(?:bonus|score|randomi[sz]ation)"
    r"[^.\n]{0,80}eliminat(?:e|es|ed|ion)",
    # attenuation coupling: attenuate ↔ separability / imprint / trace-conditioned updates
    # (verb-first)
    r"attenuat(?:e|es|ed)[^.\n]{0,80}"
    r"(?:separability|imprint|trace[-\s]conditioned\s+updates?)",
    # (noun-first)
    r"(?:separability|imprint|trace[-\s]conditioned\s+updates?)"
    r"[^.\n]{0,80}attenuat(?:e|es|ed)",
    # imprint as mechanism noun with v1 causal verbs (verb-first, inherently specific)
    r"(?:requires|driven\s+by|attributable\s+to|caus(?:ed|es|ing))"
    r"[^.\n]{0,80}imprint(?:ing)?",
    # (noun-first, inherently specific)
    r"imprint(?:ing)?"
    r"[^.\n]{0,80}(?:requires|driven\s+by|attributable\s+to|caus(?:ed|es|ing))",
    # washout as mechanism noun with v1 causal verbs — requires mechanism-context anchor
    # (verb-first: anchor before verb OR anchor after washout)
    r"(?:separabilit|probe|slate|trace|polic|state|imprint)[^.\n]{0,160}(?:requires|driven\s+by|attributable\s+to|caus(?:ed|es|ing))[^.\n]{0,80}washout"
    r"|(?:requires|driven\s+by|attributable\s+to|caus(?:ed|es|ing))[^.\n]{0,80}washout[^.\n]{0,80}(?:separabilit|probe|slate|trace|polic|state|imprint)",
    # (noun-first: anchor before washout OR anchor after verb)
    r"(?:separabilit|probe|slate|trace|polic|state|imprint)[^.\n]{0,80}washout[^.\n]{0,80}(?:requires|driven\s+by|attributable\s+to|caus(?:ed|es|ing))"
    r"|washout[^.\n]{0,80}(?:requires|driven\s+by|attributable\s+to|caus(?:ed|es|ing))[^.\n]{0,80}(?:separabilit|probe|slate|trace|polic|state|imprint)",
    # v2-era vocabulary hole — context-feature path/channel/pathway/entry,
    # mean-value path, learned-weight path/channel, selection pathway
    # (entry[-\s]point removed to its own anchor-required patterns below)
    # (verb-first)
    r"(?:requires|driven\s+by|attributable\s+to|caus(?:ed|es|ing))"
    r"[^.\n]{0,80}"
    r"(?:context[-\s]feature\s+(?:path|channel|pathway|entry)"
    r"|mean[-\s]value\s+path"
    r"|learned[-\s]weight\s+(?:path|channel)|selection\s+pathway)",
    # (noun-first)
    r"(?:context[-\s]feature\s+(?:path|channel|pathway|entry)"
    r"|mean[-\s]value\s+path"
    r"|learned[-\s]weight\s+(?:path|channel)|selection\s+pathway)"
    r"[^.\n]{0,80}(?:requires|driven\s+by|attributable\s+to|caus(?:ed|es|ing))",
    # entry-point — requires mechanism-context anchor (?:separabilit|probe|slate|mechanism)
    # Plain CLI prose ("The CLI entry-point requires three positional arguments.") has
    # no such anchor and therefore passes.
    # (verb-first: anchor before verb OR anchor after entry-point)
    r"(?:separabilit|probe|slate|mechanism)[^.\n]{0,160}(?:requires|driven\s+by|attributable\s+to|caus(?:ed|es|ing))[^.\n]{0,80}entry[-\s]point"
    r"|(?:requires|driven\s+by|attributable\s+to|caus(?:ed|es|ing))[^.\n]{0,80}entry[-\s]point[^.\n]{0,80}(?:separabilit|probe|slate|mechanism)",
    # (noun-first: anchor before entry-point OR anchor after verb)
    r"(?:separabilit|probe|slate|mechanism)[^.\n]{0,80}entry[-\s]point[^.\n]{0,80}(?:requires|driven\s+by|attributable\s+to|caus(?:ed|es|ing))"
    r"|entry[-\s]point[^.\n]{0,80}(?:requires|driven\s+by|attributable\s+to|caus(?:ed|es|ing))[^.\n]{0,80}(?:separabilit|probe|slate|mechanism)",
)

_MECHANISM_PATTERN_RES_V3: tuple[tuple[str, "re.Pattern[str]"], ...] = tuple(
    (
        pattern,
        re.compile(
            r"(?<![A-Za-z0-9])(?:" + pattern + r")(?![A-Za-z0-9])",
            re.IGNORECASE,
        ),
    )
    for pattern in MECHANISM_CLAIM_PATTERNS_V3
)

# Union of v1 + v3 compiled patterns — used by _scan_mechanism_claims.
_MECHANISM_PATTERN_RES: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    _MECHANISM_PATTERN_RES_V1 + _MECHANISM_PATTERN_RES_V3
)

# M1 hypothesis markers: any ONE licenses the sentence as hypothesis-form
# (always allowed). Matched case-insensitively as substrings of the sentence.
MECHANISM_HYPOTHESIS_MARKERS: tuple[str, ...] = (
    "hypothesize",
    "hypothesis",
    "motivates the hypothesis",
    "we test whether",
    "is consistent with",
    "may emerge",
)

# M2 scope markers: BOTH are required for a sentence to qualify as M2 form.
# Matched case-insensitively as substrings of the sentence. These mirror
# configs/prereg/axs_mechanism_prereg_v1.json scope.requiredScopeMarkers.
MECHANISM_SCOPE_MARKERS: tuple[str, ...] = (
    "within the tested policy families",
    "in this controlled testbed",
)

# Canonical tie-break caveat marker. Default used when no prereg is loaded;
# when a prereg IS loaded (CLI --prereg), the marker is read from its
# tieBreakCaveatMarker field. The caveat must appear in the M2 sentence as an
# EXACT, case-sensitive substring when the license carries caveatRequired.
DEFAULT_TIE_BREAK_CAVEAT_MARKER: str = (
    "subject to a tie-breaking sensitivity caveat (AXS-010 soft_pass)"
)

# user/privacy vocabulary co-occurring with a mechanism pattern triggers a
# Korean WARNING (stdout only, never a finding, never affects the exit code).
_MECHANISM_WARN_VOCAB_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:users?|privacy)(?![A-Za-z0-9])",
    re.IGNORECASE,
)

# Korean guidance for Track M findings (English marker strings are
# machine-read identifiers and stay English).
_TRACK_M_GUIDANCE = (
    "메커니즘 인과형 표현은 라이선스 없이 금지됨 — 가설형(M1) 문장으로 "
    "바꾸거나, scope 마커 2종('within the tested policy families' + "
    "'in this controlled testbed')과 ladder_gate M2 라이선스를 확보할 것 "
    "(caveatRequired 시 prereg tieBreakCaveatMarker 원문을 같은 문장에 포함)"
)

# Korean guidance for G-010 pattern hits (identifiers / metric names stay
# English per the logging convention). The required reframing is the
# probe-separability-diagnostic register of docs/12_CLAIM_LADDER.md.
_G010_REFRAME_GUIDANCE = (
    "leakage 개선형/프라이버시형 claim 표현은 영구 격리되어 있음 "
    "(docs/12_CLAIM_LADDER.md §4) — 'probe separability diagnostic' "
    "표현으로 재구성할 것"
)

# Approved replacement sentences (docs/12_CLAIM_LADDER.md Section 3) printed
# in the CLI guidance output when a G-010 pattern fires.
_G010_APPROVED_REPLACEMENTS: tuple[str, ...] = (
    "The expanded diagnostic removes the saturation failure observed in the "
    "earlier leakage proxy.",
    "TRACE_LIN_UCB is the only policy showing consistently positive "
    "above-null probe separability across all seed families.",
    "TRACE_GREEDY does not show positive excess probe separability under the "
    "expanded diagnostic, while maintaining strong utility.",
    "We therefore report probe separability as a diagnostic axis rather than "
    "a privacy or leakage improvement claim.",
)


class OracleNoteViolation(ValueError):
    """Raised when a post-C-014 JSON report has a missing or wrong oracleNote.

    This is a hard failure: the report carries ``oraclePolicy`` and
    ``oraclePolicyDisplayName`` (post-C-014 format) but the ``oracleNote``
    field is absent or does not equal ``REFERENCE_NOTE`` exactly.
    """


@dataclass(frozen=True)
class Finding:
    """One genuine assertion-style forbidden-claim hit.

    Attributes:
        file: path to the file the hit was found in (English).
        line: 1-based line number for phrase-match findings; ``0`` for
            structural JSON rule violations (G-008 oracle-note rule, see
            :func:`check_oracle_note`).
        phrase: the forbidden phrase that matched (from :data:`FORBIDDEN_PHRASES`
            for phrase-match findings); the sentinel ``"oracleNote"`` for
            structural JSON rule violations produced by :func:`scan_path`.
        text: the full text of the offending line (stripped of trailing newline)
            for phrase-match findings; the :class:`OracleNoteViolation` message
            string for oracle-note sentinel findings.
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


def _iter_line_matches(line: str) -> Iterable[tuple[str, re.Match]]:
    """Yield ``(label, match)`` for every forbidden phrase AND pattern hit.

    The label is the literal phrase (for :data:`FORBIDDEN_PHRASES`) or the
    regex source string (for :data:`FORBIDDEN_CLAIM_PATTERNS`, G-010). Both
    kinds of hit go through the identical suppression heuristics downstream.
    """
    for phrase in FORBIDDEN_PHRASES:
        for m in _match_iter(line, phrase):
            yield phrase, m
    for pattern, rx in _FORBIDDEN_PATTERN_RES:
        for m in rx.finditer(line):
            yield pattern, m


def _label_order(label: str) -> int:
    """Deterministic ordering index across phrases, patterns, and sentinels."""
    if label in FORBIDDEN_PHRASES:
        return FORBIDDEN_PHRASES.index(label)
    if label in FORBIDDEN_CLAIM_PATTERNS:
        return len(FORBIDDEN_PHRASES) + FORBIDDEN_CLAIM_PATTERNS.index(label)
    if label in MECHANISM_CLAIM_PATTERNS:
        return (
            len(FORBIDDEN_PHRASES)
            + len(FORBIDDEN_CLAIM_PATTERNS)
            + MECHANISM_CLAIM_PATTERNS.index(label)
        )
    if label in MECHANISM_CLAIM_PATTERNS_V3:
        return (
            len(FORBIDDEN_PHRASES)
            + len(FORBIDDEN_CLAIM_PATTERNS)
            + len(MECHANISM_CLAIM_PATTERNS)
            + MECHANISM_CLAIM_PATTERNS_V3.index(label)
        )
    return (
        len(FORBIDDEN_PHRASES)
        + len(FORBIDDEN_CLAIM_PATTERNS)
        + len(MECHANISM_CLAIM_PATTERNS)
        + len(MECHANISM_CLAIM_PATTERNS_V3)
    )


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


# ---------------------------------------------------------------------------
# G-022a — Track M mechanism-claim layer (sentence-level)
# ---------------------------------------------------------------------------

# A bullet/list item starts a fresh sentence chunk even without a blank line.
_BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s")

# Sentence terminator: ., !, ? followed by whitespace or end-of-chunk.
# Semicolons do NOT terminate a sentence (the canonical M2 sentence uses one).
_SENTENCE_END_RE = re.compile(r"[.!?](?=\s|$)")

# G-022a Fix 3: unicode evasion normalisation for the Track M sentence path.
# Zero-width chars: U+200B ZWSP, U+200C ZWNJ, U+200D ZWJ, U+FEFF BOM/ZWNBSP.
_TRACK_M_ZW_RE = re.compile(r"[​‌‍﻿]")
# Unicode dashes U+2010–U+2014 → ASCII hyphen-minus.
_TRACK_M_DASH_TABLE = str.maketrans("‐‑‒–—", "-----")


def _iter_sentence_chunks(text: str) -> Iterable[List[tuple[int, str]]]:
    """Yield chunks of ``(line_no, stripped_line)`` forming sentence groups.

    Wrapped prose lines are grouped so a sentence spanning several source
    lines is evaluated as ONE sentence. Chunk boundaries: blank lines,
    markdown headings, and bullet/list-item starts. Lines inside a
    ``## Forbidden ...`` section are skipped entirely (that section
    enumerates forbidden forms, it never asserts them).
    """
    chunk: List[tuple[int, str]] = []
    in_forbidden_section = False
    for idx, raw in enumerate(text.splitlines(), start=1):
        if _HEADING_RE.match(raw):
            in_forbidden_section = bool(
                _FORBIDDEN_SECTION_HEADING_RE.match(raw)
            )
            if chunk:
                yield chunk
                chunk = []
            continue
        stripped = raw.strip()
        if not stripped or in_forbidden_section:
            if chunk:
                yield chunk
                chunk = []
            continue
        if _BULLET_RE.match(raw) and chunk:
            yield chunk
            chunk = []
        chunk.append((idx, stripped))
    if chunk:
        yield chunk


def _split_sentences_with_offsets(joined: str) -> List[tuple[int, str]]:
    """Split ``joined`` into ``(start_offset, sentence)`` pairs."""
    out: List[tuple[int, str]] = []
    start = 0
    for m in _SENTENCE_END_RE.finditer(joined):
        end = m.end()
        if joined[start:end].strip():
            out.append((start, joined[start:end]))
        start = end
        while start < len(joined) and joined[start].isspace():
            start += 1
    if start < len(joined) and joined[start:].strip():
        out.append((start, joined[start:]))
    return out


def _scan_mechanism_claims(
    text: str,
    *,
    file: str,
    mechanism_license: Optional[dict],
    warnings: Optional[List[str]],
) -> List[Finding]:
    """Track M scan: mechanism-causal sentences forbidden unless licensed.

    Per sentence containing a mechanism pattern (identifier/backtick mentions
    suppressed): a hypothesis marker licenses it as M1 (always allowed); BOTH
    scope markers qualify it as M2-form, allowed only when ``mechanism_license``
    carries an active M2 grant (and, when ``caveatRequired``, the canonical
    caveat marker as an exact case-sensitive substring); everything else is a
    finding. ``mechanism_license=None`` means NO license (fail closed).

    When ``warnings`` is a list, a Korean warning string is appended for every
    mechanism-pattern sentence that also carries user/users/privacy vocabulary
    (warning only — never a finding).
    """
    lic = mechanism_license or {}
    m2_granted = bool(lic.get("m2", False))
    # Fail closed: a license dict without an explicit caveatRequired=False is
    # treated as caveat-required.
    caveat_required = bool(lic.get("caveatRequired", True))
    caveat_marker = lic.get("caveatMarker") or DEFAULT_TIE_BREAK_CAVEAT_MARKER

    findings: List[Finding] = []
    for chunk in _iter_sentence_chunks(text):
        # Join the chunk's lines with single spaces, remembering where each
        # source line starts so matches map back to a 1-based line number.
        starts: List[int] = []
        line_nos: List[int] = []
        parts: List[str] = []
        offset = 0
        for line_no, stripped in chunk:
            if parts:
                offset += 1  # the joining space
            starts.append(offset)
            line_nos.append(line_no)
            parts.append(stripped)
            offset += len(stripped)
        joined = " ".join(parts)

        def _line_for(abs_offset: int) -> int:
            return line_nos[max(0, bisect_right(starts, abs_offset) - 1)]

        for s_start, sentence in _split_sentences_with_offsets(joined):
            # Fix 3 (G-022a): normalise unicode evasions ONCE on a copy used
            # only for matching; line attribution still uses the original
            # sentence (reporting the sentence's first line is acceptable).
            # - Zero-width chars stripped: U+200B/C/D/FEFF
            # - Unicode dashes mapped to ASCII '-': U+2010–U+2014
            norm_sentence = _TRACK_M_ZW_RE.sub("", sentence).translate(
                _TRACK_M_DASH_TABLE
            )

            matches: List[tuple[str, re.Match]] = []
            for pattern, rx in _MECHANISM_PATTERN_RES:
                for m in rx.finditer(norm_sentence):
                    # Fix 1 (G-022a): sentence-level backtick check uses the
                    # open-span test (odd backtick count before match start)
                    # instead of the sentence-wide "backtick anywhere before
                    # AND after" heuristic used by the G-010 line scanner.
                    # The non-backtick identifier checks in _is_identifier_context
                    # are still applied via the helper; only the backtick branch
                    # is overridden here.
                    before_match = norm_sentence[: m.start()]
                    after_match = norm_sentence[m.end() :]
                    # Inside an open backtick span: odd count of backticks
                    # before the match start means the match is enclosed.
                    if before_match.count("`") % 2 == 1:
                        continue  # genuinely inside a backtick span — suppress
                    # Other identifier contexts (snake_case, camelCase, quotes)
                    # — reuse the helper but skip its backtick branch by
                    # temporarily testing a sentinel that has no backtick before.
                    # We reconstruct the effective before/after without backtick
                    # interference: strip all backticks for the helper call so
                    # only the non-backtick rules fire.
                    _before_nb = before_match.replace("`", "")
                    _after_nb = after_match.replace("`", "")
                    _line_nb = _before_nb + norm_sentence[m.start() : m.end()] + _after_nb
                    _start_nb = len(_before_nb)
                    _end_nb = _start_nb + (m.end() - m.start())
                    if _is_identifier_context(_line_nb, _start_nb, _end_nb):
                        continue  # snake_case / camelCase / quote identifier
                    matches.append((pattern, m))
            if not matches:
                continue

            first_line = _line_for(s_start + matches[0][1].start())

            # Warning rule: mechanism pattern + user/users/privacy vocabulary.
            if warnings is not None and _MECHANISM_WARN_VOCAB_RE.search(
                sentence
            ):
                warnings.append(
                    "[경고] 메커니즘 클레임 문장에 user/privacy 어휘 공존 — "
                    f"문장 재검토 권장: {file}:{first_line}"
                )

            low = sentence.lower()
            # 1. M1 hypothesis form: always allowed.
            if any(h in low for h in MECHANISM_HYPOTHESIS_MARKERS):
                continue
            # 2. M2 form (BOTH scope markers): allowed only under an active
            #    M2 license (+ exact caveat marker when caveatRequired).
            if all(s in low for s in MECHANISM_SCOPE_MARKERS):
                if m2_granted and (
                    not caveat_required or caveat_marker in sentence
                ):
                    continue
            # 3. Unlicensed mechanism claim: finding (fail closed).
            for pattern, m in matches:
                findings.append(
                    Finding(
                        file=file,
                        line=_line_for(s_start + m.start()),
                        phrase=pattern,
                        text=sentence.strip(),
                    )
                )
    return findings


# --- G-008 oracle-note rule ---


def check_oracle_note(report: dict, *, file: str = "<report>") -> None:
    """Validate the oracle-note field in a parsed JSON report (G-008 / TRD G-012).

    Rules:
    - If the report has **neither** ``oraclePolicy`` **nor**
      ``oraclePolicyDisplayName``, there is nothing to check and the function
      returns silently.
    - If the report has ``oraclePolicy`` but **no** ``oraclePolicyDisplayName``
      (legacy pre-C-014 format), a Korean warning log line is emitted but the
      function does **not** raise (legacy reports are not retroactively failed).
    - If the report has **both** ``oraclePolicy`` and
      ``oraclePolicyDisplayName`` (post-C-014 format), the ``oracleNote``
      field MUST be present and equal :data:`REFERENCE_NOTE` exactly;
      otherwise :class:`OracleNoteViolation` is raised.

    Args:
        report: parsed JSON report dict.
        file: label used in log/error messages (English identifier).

    Raises:
        OracleNoteViolation: when the post-C-014 oracle-note rule is violated.
    """
    has_oracle_policy = "oraclePolicy" in report
    has_display_name = "oraclePolicyDisplayName" in report

    if not has_oracle_policy:
        return  # no oracle policy; nothing to check.

    if not has_display_name:
        # Legacy report: emit Korean warning, do not fail.
        _logger.warning(
            "[경고] 레거시 리포트 감지: oraclePolicyDisplayName 없음 — %s "
            "(C-014 이전 형식; oracleNote 검증 생략)",
            file,
        )
        return

    # Post-C-014 format: oracleNote must equal REFERENCE_NOTE exactly.
    note = report.get("oracleNote")
    if note != REFERENCE_NOTE:
        raise OracleNoteViolation(
            f"oracleNote 불일치 — "
            f"예상: {REFERENCE_NOTE!r}, 실제: {note!r}"
        )


def scan_text(
    text: str,
    *,
    file: str = "<text>",
    mechanism_license: Optional[dict] = None,
    warnings: Optional[List[str]] = None,
) -> List[Finding]:
    """Scan ``text`` and return genuine assertion-style forbidden-claim findings.

    A line is reported for a phrase only when the phrase is used as an ASSERTION:
    it is NOT a negation/disclaimer, NOT part of an enumerated forbidden-claims
    section or denial list, and NOT part of a code identifier / config key /
    policy name. See the module docstring for the full heuristic.

    Args:
        text: the document text to scan.
        file: label recorded on each :class:`Finding` (defaults to ``<text>``).
        mechanism_license: G-022a Track M license, a dict
            ``{"m2": bool, "caveatRequired": bool, "caveatMarker": str}``.
            ``None`` (default) means NO license is active — M2-form mechanism
            sentences are findings (fail closed).
        warnings: optional list; Korean Track M warning strings are appended
            (mechanism pattern + user/privacy vocabulary). Warnings are never
            findings and never affect the exit code.

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

        for phrase, m in _iter_line_matches(line):
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

    # G-022a: Track M mechanism-claim layer (sentence-level, fail closed).
    findings.extend(
        _scan_mechanism_claims(
            text,
            file=file,
            mechanism_license=mechanism_license,
            warnings=warnings,
        )
    )

    findings.sort(key=lambda f: (f.line, f.text, _label_order(f.phrase)))
    return findings


def scan_path(
    path: str | Path,
    *,
    mechanism_license: Optional[dict] = None,
    warnings: Optional[List[str]] = None,
) -> List[Finding]:
    """Scan a single file and return its genuine findings.

    For JSON files the function additionally runs the G-008 oracle-note rule
    via :func:`check_oracle_note`. A post-C-014 violation is converted to a
    :class:`Finding` with ``phrase="oracleNote"`` and ``line=0``; a legacy
    warning is logged but produces no Finding. Non-text / unreadable files are
    skipped (returns an empty list). The file path string is recorded on each
    finding.
    """
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    findings = scan_text(
        text,
        file=str(p),
        mechanism_license=mechanism_license,
        warnings=warnings,
    )

    # G-008: run oracle-note JSON rule on .json files.
    if p.suffix.lower() == ".json":
        try:
            report = json.loads(text)
        except json.JSONDecodeError:
            report = None
        if isinstance(report, dict):
            try:
                check_oracle_note(report, file=str(p))
            except OracleNoteViolation as exc:
                findings.append(
                    Finding(
                        file=str(p),
                        line=0,
                        phrase="oracleNote",
                        text=str(exc),
                    )
                )

    return findings


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


def scan_paths(
    paths: Sequence[str | Path],
    *,
    mechanism_license: Optional[dict] = None,
    warnings: Optional[List[str]] = None,
) -> List[Finding]:
    """Scan multiple files/directories and return all genuine findings.

    Directories are walked recursively for ``.md`` / ``.json`` / ``.txt`` files.
    Findings are returned in a deterministic order (by file, then line).
    ``mechanism_license`` / ``warnings`` are forwarded to :func:`scan_text`
    (G-022a Track M; ``None`` license = fail closed).
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
            findings.extend(
                scan_path(
                    f,
                    mechanism_license=mechanism_license,
                    warnings=warnings,
                )
            )
    findings.sort(key=lambda f: (f.file, f.line, f.text))
    return findings


def _default_targets() -> List[Path]:
    """Return the default scan targets: ``docs/`` and ``outputs/reports/``."""
    repo_root = Path(__file__).resolve().parents[3]
    return [repo_root / "docs", repo_root / "outputs" / "reports"]


def _resolve_mechanism_license(args: argparse.Namespace) -> Optional[dict]:
    """Resolve the Track M M2 license from gate arguments (G-022a).

    The license is recomputed ONCE via
    :func:`echo_bench.tools.ladder_gate.evaluate_mechanism_license` when ALL
    of ``--prereg`` / ``--reports`` / ``--ledger`` are provided. Any other
    state — args missing, gate error — returns ``None`` (no license, fail
    closed). licenses.json is NEVER read.
    """
    if not (args.prereg and args.reports and args.ledger):
        if args.prereg or args.reports or args.ledger:
            print(
                "[경고] --prereg/--reports/--ledger 는 셋 모두 함께 지정해야 함 "
                "— M2 라이선스 미부여 (fail closed)"
            )
        return None

    import echo_bench.tools.ladder_gate as _ladder_gate

    try:
        gate = _ladder_gate.evaluate_mechanism_license(
            args.prereg,
            args.reports,
            ledger_path=args.ledger,
            release=args.release,
        )
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(
            f"[경고] ladder_gate 평가 실패 — M2 라이선스 미부여 (fail closed): {exc}"
        )
        return None

    # Canonical caveat marker: read from the loaded prereg; fall back to the
    # committed default only if the field is absent.
    caveat_marker = DEFAULT_TIE_BREAK_CAVEAT_MARKER
    try:
        with open(args.prereg, "r", encoding="utf-8") as fh:
            prereg = json.load(fh)
        marker = prereg.get("tieBreakCaveatMarker")
        if isinstance(marker, str) and marker:
            caveat_marker = marker
    except (OSError, json.JSONDecodeError, AttributeError):
        pass

    return {
        "m2": bool((gate.get("rungs") or {}).get("M2", False)),
        "caveatRequired": bool(gate.get("caveatRequired", True)),
        "caveatMarker": caveat_marker,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Scan targets, print findings, return an exit code.

    Args:
        argv: optional arguments: scan paths (when empty, scans the default
            targets ``docs/`` and ``outputs/reports/``) plus the optional
            Track M gate arguments ``--prereg`` / ``--reports`` / ``--ledger``
            / ``--release`` (all three evidence args required together;
            otherwise no M2 license is active — fail closed).

    Returns:
        ``0`` when the scanned tree is clean; ``1`` when any genuine
        assertion-style forbidden claim is found. Track M warnings never
        affect the exit code.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="claim_check",
        description="ECHO-Bench 금지 클레임 스캐너 (G-005/G-008/G-010/G-022a)",
    )
    parser.add_argument("paths", nargs="*", help="스캔 대상 경로 목록")
    parser.add_argument("--prereg", default=None, help="사전등록 JSON 경로")
    parser.add_argument(
        "--reports", nargs="*", default=None, help="실험 리포트 JSON 경로 목록"
    )
    parser.add_argument("--ledger", default=None, help="실행 원장 JSON 경로")
    parser.add_argument(
        "--release",
        action="store_true",
        default=False,
        help="원격 ancestry 추가 확인 (ladder_gate 로 전달)",
    )
    args = parser.parse_args(argv)

    targets: Sequence[str | Path] = (
        args.paths if args.paths else _default_targets()
    )

    mechanism_license = _resolve_mechanism_license(args)

    track_m_warnings: List[str] = []
    findings = scan_paths(
        targets,
        mechanism_license=mechanism_license,
        warnings=track_m_warnings,
    )

    # Track M warnings: stdout only, NEVER findings, NEVER affect exit code.
    for w in track_m_warnings:
        print(w)

    if findings:
        for f in findings:
            # Identifiers / paths stay English.
            print(f"{f.file}:{f.line}: forbidden claim '{f.phrase}': {f.text}")
        # G-010: Korean guidance for leakage-improvement / privacy claim-form
        # pattern hits — name the pattern and the required reframing.
        g010_hits = [f for f in findings if f.phrase in FORBIDDEN_CLAIM_PATTERNS]
        if g010_hits:
            for f in g010_hits:
                print(
                    f"[G-010] 금지 패턴 '{f.phrase}' 감지 "
                    f"({f.file}:{f.line}) — {_G010_REFRAME_GUIDANCE}"
                )
            print(
                "[G-010] 승인된 대체 문장 "
                "(probe separability diagnostic 표현, "
                "docs/12_CLAIM_LADDER.md §3):"
            )
            for sentence in _G010_APPROVED_REPLACEMENTS:
                print(f"  - {sentence}")
        # G-022a: Korean guidance for Track M mechanism-claim hits (v1 + v3).
        _all_mechanism_patterns = set(MECHANISM_CLAIM_PATTERNS) | set(
            MECHANISM_CLAIM_PATTERNS_V3
        )
        track_m_hits = [
            f for f in findings if f.phrase in _all_mechanism_patterns
        ]
        if track_m_hits:
            for f in track_m_hits:
                print(
                    f"[Track M] 금지 메커니즘 클레임 패턴 '{f.phrase}' 감지 "
                    f"({f.file}:{f.line}) — {_TRACK_M_GUIDANCE}"
                )
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

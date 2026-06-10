"""Tests for the G-005 context-aware forbidden-claim scanner.

Covers the acceptance criteria:

(a) a real assertion-style forbidden claim is FLAGGED;
(b) disclaimers / negations are NOT flagged;
(c) the docs/01_GUARDRAILS.md "Forbidden Claims" enumeration is NOT flagged;
(d) the identifier PSEUDO_USER_MODEL / key "user_model" is NOT flagged;
(e) the live docs/ + outputs/reports/ tree returns NO genuine findings.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from echo_bench.tools.claim_check import (
    FORBIDDEN_PHRASES,
    Finding,
    main,
    scan_path,
    scan_paths,
    scan_text,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DOCS_DIR = _REPO_ROOT / "docs"
_REPORTS_DIR = _REPO_ROOT / "outputs" / "reports"


# --------------------------------------------------------------------------- #
# (a) Real assertion-style forbidden claims ARE flagged.
# --------------------------------------------------------------------------- #


def test_real_claim_is_flagged():
    findings = scan_text("Users prefer this layout over the alternative.")
    assert findings, "an assertion-style 'users prefer' claim must be flagged"
    assert findings[0].phrase == "users prefer"
    assert findings[0].line == 1


@pytest.mark.parametrize(
    "text, phrase",
    [
        ("The system improves user wellbeing for everyone.", "wellbeing"),
        ("This layout has therapeutic value for the user.", "therapeutic"),
        ("The benchmark is GDPR compliant across regions.", "gdpr compliant"),
        ("Each card reflects the user's personality traits.", "personality"),
        ("Users understand the slate at a glance.", "users understand"),
        ("The stimuli are meaning-free shapes.", "meaning-free"),
    ],
)
def test_various_real_claims_flagged(text, phrase):
    findings = scan_text(text)
    assert any(f.phrase == phrase for f in findings), (
        f"expected phrase {phrase!r} to be flagged in {text!r}"
    )


def test_finding_fields_are_populated():
    findings = scan_text("Users prefer this.\nAnother line.", file="demo.md")
    assert len(findings) == 1
    f = findings[0]
    assert isinstance(f, Finding)
    assert f.file == "demo.md"
    assert f.line == 1
    assert f.phrase == "users prefer"
    assert "Users prefer this." == f.text


# --------------------------------------------------------------------------- #
# (b) Disclaimers / negations are NOT flagged.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "This is NOT a privacy guarantee.",
        "The leakage metric is not an emotion or wellbeing judgment.",
        "salienceScore is not an aesthetic, preference, emotion, or wellbeing judgment.",
        "No claim about user preference, emotion, or wellbeing is made.",
        "The schema forbids any user / persona / emotion / preference field.",
        "These metrics are not measures of personality or diagnosis.",
        "The testbed makes no wellbeing claim whatsoever.",
        "It represents nothing about persona, emotion, or wellbeing.",
    ],
)
def test_disclaimers_not_flagged(text):
    assert scan_text(text) == [], f"disclaimer wrongly flagged: {text!r}"


def test_multiline_wrapped_disclaimer_not_flagged():
    text = (
        "These metrics are not measures\n"
        "of user preference, understanding, emotion, wellbeing, or anonymity."
    )
    assert scan_text(text) == []


def test_wrapped_list_tail_not_flagged():
    text = (
        "No policy may read or emit a user_id, persona, preference vector, or\n"
        "diagnosis label."
    )
    assert scan_text(text) == []


# --------------------------------------------------------------------------- #
# (c) The guardrails "Forbidden Claims" enumeration is NOT flagged.
# --------------------------------------------------------------------------- #


def test_forbidden_claims_section_not_flagged():
    text = (
        "# Guardrails\n"
        "\n"
        "## Forbidden Claims\n"
        "- Users prefer this\n"
        "- Users understand this\n"
        "- Emotion, mood, personality, wellbeing\n"
        "- Legal/GDPR compliance\n"
        "\n"
        "## Core Rule\n"
        "The system adapts over traces, not users.\n"
    )
    assert scan_text(text) == []


def test_live_guardrails_doc_not_flagged():
    findings = scan_path(_DOCS_DIR / "01_GUARDRAILS.md")
    assert findings == [], f"01_GUARDRAILS.md wrongly flagged: {findings}"


# --------------------------------------------------------------------------- #
# (d) Identifiers / config keys / policy names are NOT flagged.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "The PSEUDO_USER_MODEL policy is the only one with a latent vector.",
        'The trace rejects the "user_model" key.',
        "The forbidden field preference_vector never enters the schema.",
        "Report the salience_outlier_rate per policy.",
        "The `emotion` key is rejected on load.",
        "`persona` and `preference` are forbidden-field tokens.",
        "FORBIDDEN_FIELDS includes user_model and preference.",
    ],
)
def test_identifiers_not_flagged(text):
    assert scan_text(text) == [], f"identifier wrongly flagged: {text!r}"


def test_pseudo_user_model_word_boundary():
    # The policy name must not be flagged even though it embeds 'user'/'model'.
    findings = scan_text("PSEUDO_USER_MODEL exists purely as a contrast baseline.")
    assert findings == []


# --------------------------------------------------------------------------- #
# (e) The live docs/ + outputs/reports/ tree is clean.
# --------------------------------------------------------------------------- #


def test_live_tree_is_clean():
    findings = scan_paths([_DOCS_DIR, _REPORTS_DIR])
    assert findings == [], (
        "live docs/ + outputs/reports/ produced genuine forbidden-claim "
        f"findings (report, do not weaken the scanner): {findings}"
    )


def test_methods_doc_is_clean():
    findings = scan_path(_DOCS_DIR / "methods.md")
    assert findings == [], f"methods.md produced findings: {findings}"


# --------------------------------------------------------------------------- #
# CLI behaviour.
# --------------------------------------------------------------------------- #


def test_cli_clean_exits_zero(capsys):
    rc = main([str(_DOCS_DIR), str(_REPORTS_DIR)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "통과" in out  # Korean pass summary


def test_cli_dirty_exits_nonzero(tmp_path, capsys):
    dirty = tmp_path / "dirty.md"
    dirty.write_text("Users prefer this layout.\n", encoding="utf-8")
    rc = main([str(dirty)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "users prefer" in out
    assert "실패" in out  # Korean fail summary


def test_default_targets_exist():
    # main() with no argv must scan the default docs/ + reports/ and pass.
    rc = main([])
    assert rc == 0


def test_phrase_list_matches_sources():
    # Sanity: the validate-claims source phrases are all present.
    for p in (
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
    ):
        assert p in FORBIDDEN_PHRASES

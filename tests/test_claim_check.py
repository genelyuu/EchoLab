"""Tests for the G-005 context-aware forbidden-claim scanner.

Covers the acceptance criteria:

(a) a real assertion-style forbidden claim is FLAGGED;
(b) disclaimers / negations are NOT flagged;
(c) the docs/01_GUARDRAILS.md "Forbidden Claims" enumeration is NOT flagged;
(d) the identifier PSEUDO_USER_MODEL / key "user_model" is NOT flagged;
(e) the live docs/ + outputs/reports/ tree returns NO genuine findings.

G-008 / TRD G-012 additions:
(f) oracle-terminology phrases ("global upper bound", "global optimum",
    "globally optimal") are FLAGGED as assertion-style claims;
(g) negated forms (e.g. the exact REFERENCE_NOTE string) are NOT flagged;
(h) JSON oracle-note rule: post-C-014 reports with wrong/missing oracleNote
    fail; legacy reports (no oraclePolicyDisplayName) only warn; compliant
    reports pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from echo_bench.tools.claim_check import (
    FORBIDDEN_PHRASES,
    Finding,
    main,
    scan_path,
    scan_paths,
    scan_text,
    check_oracle_note,
    OracleNoteViolation,
)
from echo_bench.policies.display_names import REFERENCE_NOTE

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


# --------------------------------------------------------------------------- #
# (f) G-008: Oracle-terminology phrases are FLAGGED.                          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text, phrase",
    [
        (
            "The oracle provides a global upper bound on performance.",
            "global upper bound",
        ),
        (
            "ORACLE_STRATEGY achieves the global optimum for coverage.",
            "global optimum",
        ),
        (
            "The reference policy is globally optimal.",
            "globally optimal",
        ),
        (
            "This policy reaches a global upper bound across seeds.",
            "global upper bound",
        ),
    ],
)
def test_oracle_terminology_flagged(text, phrase):
    findings = scan_text(text)
    assert any(f.phrase == phrase for f in findings), (
        f"expected oracle phrase {phrase!r} to be flagged in {text!r}"
    )


def test_oracle_phrases_in_forbidden_list():
    for p in ("global upper bound", "global optimum", "globally optimal"):
        assert p in FORBIDDEN_PHRASES, f"{p!r} must be in FORBIDDEN_PHRASES"


# --------------------------------------------------------------------------- #
# (g) G-008: Negated oracle-terminology is NOT flagged.                       #
# --------------------------------------------------------------------------- #


def test_reference_note_not_flagged():
    # The exact REFERENCE_NOTE constant must never be flagged (it contains
    # "not global optimum" which the existing negator suppression should catch).
    assert scan_text(REFERENCE_NOTE) == [], (
        f"REFERENCE_NOTE must not be flagged: {REFERENCE_NOTE!r}"
    )


@pytest.mark.parametrize(
    "text",
    [
        # Various negated forms that must not be flagged.
        "This is not a global upper bound.",
        "The oracle is not globally optimal.",
        "No global optimum is claimed here.",
        "The policy never reaches global optimum.",
        "objective-specific reference, not global optimum",
        "This reference is not the global optimum for all tasks.",
        "These are objective-specific references, not global upper bounds.",
    ],
)
def test_negated_oracle_terminology_not_flagged(text):
    assert scan_text(text) == [], (
        f"negated oracle terminology must not be flagged: {text!r}"
    )


# --------------------------------------------------------------------------- #
# (h) G-008: JSON oracle-note rule via check_oracle_note.                     #
# --------------------------------------------------------------------------- #


def _make_report(*, oracle_policy=None, display_name=None, note=None):
    """Build a minimal report dict for oracle-note tests."""
    d = {}
    if oracle_policy is not None:
        d["oraclePolicy"] = oracle_policy
    if display_name is not None:
        d["oraclePolicyDisplayName"] = display_name
    if note is not None:
        d["oracleNote"] = note
    return d


def test_oracle_note_compliant_passes():
    report = _make_report(
        oracle_policy="ORACLE_STRATEGY",
        display_name="STRATEGY_OBJECTIVE_REFERENCE",
        note=REFERENCE_NOTE,
    )
    # Must not raise.
    check_oracle_note(report, file="test.json")


def test_oracle_note_wrong_value_raises():
    report = _make_report(
        oracle_policy="ORACLE_STRATEGY",
        display_name="STRATEGY_OBJECTIVE_REFERENCE",
        note="this is the global optimum",
    )
    with pytest.raises(OracleNoteViolation):
        check_oracle_note(report, file="test.json")


def test_oracle_note_missing_raises():
    report = _make_report(
        oracle_policy="ORACLE_STRATEGY",
        display_name="STRATEGY_OBJECTIVE_REFERENCE",
        # no note key
    )
    with pytest.raises(OracleNoteViolation):
        check_oracle_note(report, file="test.json")


def test_oracle_note_legacy_no_display_name_warns_not_raises():
    """Legacy report (oraclePolicy present, oraclePolicyDisplayName absent) => warning only."""
    from unittest.mock import patch
    import echo_bench.tools.claim_check as cc_mod

    report = _make_report(oracle_policy="ORACLE_STRATEGY")
    # Patch _logger.warning so we can assert it was called even though
    # the logger has propagate=False (which prevents caplog from capturing).
    with patch.object(cc_mod._logger, "warning") as mock_warn:
        check_oracle_note(report, file="legacy.json")  # must NOT raise
    # A warning log must have been emitted (Korean text expected).
    mock_warn.assert_called_once()
    # Both the Korean legacy marker AND the file label must appear in the call.
    call_args = mock_warn.call_args
    assert "레거시" in str(call_args), (
        "warning must contain the Korean legacy marker '레거시'"
    )
    assert "legacy.json" in str(call_args), (
        "warning must reference the file label 'legacy.json'"
    )


def test_oracle_note_no_oracle_policy_passes():
    """Report without oraclePolicy key: nothing to check, passes silently."""
    report = {"experiment": "e2", "table": []}
    check_oracle_note(report, file="no_oracle.json")  # must not raise


def test_oracle_note_scan_on_compliant_report_json(tmp_path):
    """scan_path on a compliant post-C-014 report JSON exits cleanly."""
    report = _make_report(
        oracle_policy="ORACLE_COVERAGE",
        display_name="COVERAGE_GREEDY_REFERENCE",
        note=REFERENCE_NOTE,
    )
    p = tmp_path / "e2_compliant.json"
    p.write_text(json.dumps(report), encoding="utf-8")
    findings = scan_path(p)
    assert findings == [], f"compliant report should not produce findings: {findings}"


def test_oracle_note_scan_on_bad_report_json_flagged(tmp_path):
    """scan_path on a post-C-014 report with bad oracleNote produces a Finding."""
    report = _make_report(
        oracle_policy="ORACLE_COVERAGE",
        display_name="COVERAGE_GREEDY_REFERENCE",
        note="wrong note — global upper bound",
    )
    p = tmp_path / "e2_bad.json"
    p.write_text(json.dumps(report), encoding="utf-8")
    findings = scan_path(p)
    assert findings, "bad oracleNote in post-C-014 report must produce a Finding"
    assert any(f.phrase == "oracleNote" for f in findings)


def test_oracle_note_scan_on_legacy_report_json_clean(tmp_path):
    """scan_path on a legacy report (no oraclePolicyDisplayName) => no Finding."""
    from unittest.mock import patch
    import echo_bench.tools.claim_check as cc_mod

    report = _make_report(oracle_policy="ORACLE_COVERAGE")
    p = tmp_path / "e2_legacy.json"
    p.write_text(json.dumps(report), encoding="utf-8")
    with patch.object(cc_mod._logger, "warning"):
        findings = scan_path(p)
    assert findings == [], "legacy report must not produce a Finding"


# --------------------------------------------------------------------------- #
# (i) G-008: Live reports directory is clean.                                 #
# --------------------------------------------------------------------------- #


def test_live_reports_oracle_notes_clean():
    """All post-C-014 E2 reports in outputs/reports/ have correct oracleNote."""
    findings = scan_paths([_REPORTS_DIR])
    oracle_findings = [f for f in findings if f.phrase == "oracleNote"]
    assert oracle_findings == [], (
        f"live reports/ have oracle-note violations: {oracle_findings}"
    )

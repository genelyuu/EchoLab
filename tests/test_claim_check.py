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

G-010 additions (claim ladder v2):
(j) leakage-improvement / privacy claim-FORM regex patterns are FLAGGED;
(k) backtick-quoted (mention) and negated forms are NOT flagged — the ladder's
    own forbidden-forms list must stay scanner-safe;
(l) the CLI emits a Korean guidance message naming the pattern and the
    required "probe separability diagnostic" reframing, with the approved
    replacement sentences;
(m) docs/12_CLAIM_LADDER.md carries ladderVersion claim-ladder-2, the five
    tracks (U/S/N/R/G), the exact Track S activation sentences (EN + KR), and
    the "(formerly Track L)" legacy note, and scans clean.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from echo_bench.tools.claim_check import (
    FORBIDDEN_CLAIM_PATTERNS,
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


# --------------------------------------------------------------------------- #
# (j) G-010: leakage-improvement / privacy claim-FORM patterns are FLAGGED.   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text, pattern",
    [
        ("TRACE_GREEDY reduces leakage.", r"reduce(?:s|d)?\s+leakage"),
        (
            "The policy reduced leakage at every horizon.",
            r"reduce(?:s|d)?\s+leakage",
        ),
        ("Trace-only policies reduce leakage.", r"reduce(?:s|d)?\s+leakage"),
        ("TRACE_GREEDY improves leakage.", r"improv(?:e|es|ed|ing)\s+leakage"),
        (
            "The bandit update improved leakage across families.",
            r"improv(?:e|es|ed|ing)\s+leakage",
        ),
        (
            "TRACE_LIN_UCB leaks user information.",
            r"leaks?\s+user\s+information",
        ),
        ("The system is privacy-preserving.", r"privacy[-\s]preserving"),
        ("Probe separability is privacy leakage.", r"is\s+privacy\s+leakage"),
    ],
)
def test_g010_claim_form_patterns_flagged(text, pattern):
    findings = scan_text(text)
    assert any(f.phrase == pattern for f in findings), (
        f"expected pattern {pattern!r} to be flagged in {text!r}: {findings}"
    )


def test_g010_patterns_in_pattern_list():
    for p in (
        r"reduce(?:s|d)?\s+leakage",
        r"improv(?:e|es|ed|ing)\s+leakage",
        r"leaks?\s+user\s+information",
        r"privacy[-\s]preserving",
        r"is\s+privacy\s+leakage",
    ):
        assert p in FORBIDDEN_CLAIM_PATTERNS, (
            f"{p!r} must be in FORBIDDEN_CLAIM_PATTERNS"
        )


# --------------------------------------------------------------------------- #
# (k) G-010: backtick-quoted (mention) and negated forms are NOT flagged.     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        # Backtick code-span quoting = mention, not assertion (the ladder's own
        # forbidden-forms list uses exactly this mechanism).
        "- `TRACE_GREEDY reduces leakage.`",
        "- `TRACE_GREEDY improves leakage.`",
        "- `TRACE_LIN_UCB leaks user information.`",
        "- `Probe separability is privacy leakage.`",
        "- `The system is privacy-preserving.`",
        "Statements like `trace-only policies reduce leakage` are quarantined.",
        # Negated / denial forms.
        "TRACE_GREEDY does not reduce leakage in any admissible sense.",
        "No policy may be described as privacy-preserving.",
        'NEVER write "randomized policies leak user information".',
        "This is not a privacy-preserving claim and never will be.",
    ],
)
def test_g010_quoted_or_negated_forms_not_flagged(text):
    assert scan_text(text) == [], (
        f"quoted/negated claim form wrongly flagged: {text!r}"
    )


# --------------------------------------------------------------------------- #
# (l) G-010: CLI Korean guidance names the pattern and the reframing.         #
# --------------------------------------------------------------------------- #


def test_g010_cli_korean_guidance(tmp_path, capsys):
    dirty = tmp_path / "dirty.md"
    dirty.write_text("TRACE_GREEDY reduces leakage.\n", encoding="utf-8")
    rc = main([str(dirty)])
    out = capsys.readouterr().out
    assert rc == 1
    # The offending pattern is named.
    assert r"reduce(?:s|d)?\s+leakage" in out
    # Korean guidance with the required reframing.
    assert "재구성" in out
    assert "probe separability diagnostic" in out
    # Approved replacement sentences are documented in the guidance output.
    assert (
        "We therefore report probe separability as a diagnostic axis rather "
        "than a privacy or leakage improvement claim." in out
    )


def test_g010_cli_no_guidance_when_clean(tmp_path, capsys):
    clean = tmp_path / "clean.md"
    clean.write_text(
        "TRACE_LIN_UCB exhibits above-null probe separability.\n",
        encoding="utf-8",
    )
    rc = main([str(clean)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "[G-010]" not in out


# --------------------------------------------------------------------------- #
# (m) G-010: ladder v2 structure — version, tracks, Track S sentences.        #
# --------------------------------------------------------------------------- #

_LADDER_PATH = _DOCS_DIR / "12_CLAIM_LADDER.md"


def test_ladder_version_is_2():
    text = _LADDER_PATH.read_text(encoding="utf-8")
    assert "ladderVersion: claim-ladder-2" in text
    # claim-ladder-1 may appear only in the revision history, never as the
    # current ladderVersion declaration.
    assert "ladderVersion: claim-ladder-1" not in text


def test_ladder_has_five_tracks():
    text = _LADDER_PATH.read_text(encoding="utf-8")
    for track in ("Track U", "Track S", "Track N", "Track R", "Track G"):
        assert track in text, f"ladder must define {track}"


def test_ladder_track_s_activation_sentences_exact():
    text = _LADDER_PATH.read_text(encoding="utf-8")
    en = (
        "Track S is activated when a policy exhibits positive, "
        "cross-family-consistent, null-corrected probe separability. "
        "This is a diagnostic signal, not an improvement claim."
    )
    kr = (
        "Track S는 policy가 null-corrected probe separability를 "
        "cross-family 일관적으로 양수로 보일 때 활성화된다. "
        "이는 진단 신호이지 개선 claim이 아니다."
    )
    normalized = " ".join(text.split())
    assert " ".join(en.split()) in normalized, (
        "ladder must contain the exact Track S activation sentence (EN)"
    )
    assert " ".join(kr.split()) in normalized, (
        "ladder must contain the exact Track S activation sentence (KR)"
    )


def test_ladder_keeps_legacy_track_l_note():
    text = _LADDER_PATH.read_text(encoding="utf-8")
    assert "formerly Track L" in text


def test_ladder_records_activation_decision_and_evidence():
    text = _LADDER_PATH.read_text(encoding="utf-8")
    assert "leakage_diagnostic_a72a1c0582fe" in text
    assert "c3550dc423d7" in text
    assert "TRACE_LIN_UCB" in text


def test_ladder_scans_clean():
    findings = scan_path(_LADDER_PATH)
    assert findings == [], (
        f"revised ladder must pass its own scanner: {findings}"
    )


# --------------------------------------------------------------------------- #
# (n) G-022a/G-022c: Track M mechanism-claim layer.                            #
# --------------------------------------------------------------------------- #

from echo_bench.tools.claim_check import (  # noqa: E402
    DEFAULT_TIE_BREAK_CAVEAT_MARKER,
    MECHANISM_CLAIM_PATTERNS,
)

_PAT_REQUIRES = r"requires\s+history[-\s]dependent\s+exploration"
_PAT_ATTRIBUTABLE = (
    r"attributable\s+to\s+(?:the\s+)?(?:history[-\s]dependent\s+)?exploration"
)
_PAT_DRIVEN = r"driven\s+by\s+(?:the\s+)?(?:history[-\s]dependent\s+)?exploration"
_PAT_CAUSES = r"caus(?:ed|es|ing)\s+[^.\n]{0,80}probe[-\s]separability"
_PAT_NOT_ADAPTIVITY = r"not\s+adaptivity\s+itself"

_CAVEAT = "subject to a tie-breaking sensitivity caveat (AXS-010 soft_pass)"

_M2_CANONICAL = (
    "Within the tested policy families in this controlled testbed, above-null "
    "slate separability requires history-dependent exploration pressure "
    "coupled to trace-conditioned update dynamics; trace-independent "
    "randomization and schedule-matched bonuses do not produce the same "
    "effect."
)
# Same single sentence with the canonical caveat marker spliced in before the
# final period (so the caveat is part of the SAME sentence).
_M2_WITH_CAVEAT = _M2_CANONICAL[:-1] + ", " + _CAVEAT + "."
_M2_WITH_PARAPHRASED_CAVEAT = (
    _M2_CANONICAL[:-1] + ", noting minor ordering effects."
)


def test_mechanism_pattern_list_exact():
    # EXACTLY the five preregistered patterns — no broadening, no additions.
    assert MECHANISM_CLAIM_PATTERNS == (
        _PAT_REQUIRES,
        _PAT_ATTRIBUTABLE,
        _PAT_DRIVEN,
        _PAT_CAUSES,
        _PAT_NOT_ADAPTIVITY,
    )


def test_default_caveat_marker_matches_prereg():
    prereg = json.loads(
        (_REPO_ROOT / "configs" / "prereg" / "axs_mechanism_prereg_v1.json")
        .read_text(encoding="utf-8")
    )
    assert DEFAULT_TIE_BREAK_CAVEAT_MARKER == prereg["tieBreakCaveatMarker"]
    assert DEFAULT_TIE_BREAK_CAVEAT_MARKER == _CAVEAT


# --- PASS fixtures (no finding) ---


@pytest.mark.parametrize(
    "text",
    [
        # Hypothesis form (M1) — always allowed.
        "We hypothesize that above-null slate separability may emerge when "
        "history-dependent exploration pressure is coupled to "
        "trace-conditioned updates.",
        "Current results motivate a mechanism hypothesis: above-null slate "
        "separability may emerge when history-dependent exploration pressure "
        "is coupled to trace-conditioned updates.",
    ],
)
def test_mechanism_hypothesis_sentences_allowed(text):
    assert scan_text(text) == [], f"M1 hypothesis form wrongly flagged: {text!r}"


def test_mechanism_m2_sentence_allowed_with_license_no_caveat():
    findings = scan_text(
        _M2_CANONICAL,
        mechanism_license={"m2": True, "caveatRequired": False},
    )
    assert findings == [], f"licensed M2 sentence wrongly flagged: {findings}"


def test_mechanism_m2_sentence_allowed_with_license_and_caveat():
    findings = scan_text(
        _M2_WITH_CAVEAT,
        mechanism_license={"m2": True, "caveatRequired": True},
    )
    assert findings == [], (
        f"licensed M2 sentence with canonical caveat wrongly flagged: {findings}"
    )


def test_requires_the_exploration_config_key_no_match():
    # "requires the exploration" != "requires history-dependent exploration":
    # the sentence must not match ANY mechanism pattern.
    text = "The runner requires the exploration config key."
    assert scan_text(text) == []
    import re as _re

    for pat in MECHANISM_CLAIM_PATTERNS:
        assert _re.search(pat, text, _re.IGNORECASE) is None, (
            f"pattern {pat!r} must not match {text!r}"
        )


@pytest.mark.parametrize(
    "text",
    [
        # snake_case identifier: the regex itself cannot match underscores.
        "Set the `requires_history_dependent_exploration` flag in the config.",
        # backtick code-span mention of the claim FORM (mention, not assertion).
        "The form `requires history-dependent exploration` is a quarantined "
        "mechanism claim form.",
    ],
)
def test_mechanism_identifier_context_suppressed(text):
    assert scan_text(text) == [], f"identifier context wrongly flagged: {text!r}"


# --- FAIL fixtures (finding) ---


@pytest.mark.parametrize(
    "text, pattern",
    [
        (
            "Amplification is caused by exploration-driven probe separability.",
            _PAT_CAUSES,
        ),
        (
            "Amplification generally requires history-dependent exploration.",
            _PAT_REQUIRES,
        ),
        ("The effect is attributable to exploration.", _PAT_ATTRIBUTABLE),
    ],
)
def test_mechanism_unmarked_causal_sentences_flagged(text, pattern):
    findings = scan_text(text)
    assert any(f.phrase == pattern for f in findings), (
        f"expected mechanism pattern {pattern!r} flagged in {text!r}: {findings}"
    )


def test_mechanism_driven_by_not_adaptivity_flags_both_patterns():
    text = "The amplification is driven by exploration, not adaptivity itself."
    findings = scan_text(text)
    phrases = {f.phrase for f in findings}
    assert _PAT_DRIVEN in phrases, findings
    assert _PAT_NOT_ADAPTIVITY in phrases, findings


def test_mechanism_m2_sentence_without_license_flagged():
    # Fail closed: no gate evidence -> M2-form sentences are findings.
    findings = scan_text(_M2_CANONICAL)  # mechanism_license=None
    assert findings, "M2-form sentence without license must be a finding"
    assert all(f.phrase in MECHANISM_CLAIM_PATTERNS for f in findings)


def test_mechanism_m2_sentence_with_denied_license_flagged():
    findings = scan_text(_M2_CANONICAL, mechanism_license={"m2": False})
    assert findings, "M2-form sentence with m2=False must be a finding"


def test_mechanism_m2_caveat_required_but_absent_flagged():
    findings = scan_text(
        _M2_CANONICAL,
        mechanism_license={"m2": True, "caveatRequired": True},
    )
    assert findings, "caveatRequired=True without the caveat must be a finding"


def test_mechanism_m2_paraphrased_caveat_flagged():
    findings = scan_text(
        _M2_WITH_PARAPHRASED_CAVEAT,
        mechanism_license={"m2": True, "caveatRequired": True},
    )
    assert findings, "paraphrased caveat must NOT satisfy the caveat rule"


def test_mechanism_m2_caveat_marker_is_case_sensitive():
    upper = _M2_CANONICAL[:-1] + ", " + _CAVEAT.upper() + "."
    findings = scan_text(
        upper,
        mechanism_license={"m2": True, "caveatRequired": True},
    )
    assert findings, "case-mangled caveat marker must NOT satisfy the rule"


def test_mechanism_single_scope_marker_flagged_even_with_license():
    # Only ONE of the two required scope markers -> finding, license or not.
    text = (
        "In this controlled testbed, above-null slate separability requires "
        "history-dependent exploration pressure."
    )
    findings = scan_text(
        text, mechanism_license={"m2": True, "caveatRequired": False}
    )
    assert findings, "single scope marker must not qualify as M2 form"


# --- WARNING fixtures (stdout warning, never affects findings/exit) ---

_WARNING_SENTENCE = (
    "Within the tested policy families in this controlled testbed, "
    "exploration pressure requires history-dependent exploration and may "
    "affect users."
)


def test_mechanism_user_vocab_warning_collected():
    warnings: list = []
    findings = scan_text(_WARNING_SENTENCE, file="warn.md", warnings=warnings)
    assert len(warnings) == 1, warnings
    assert "[경고]" in warnings[0]
    assert "warn.md:1" in warnings[0]
    # The finding/license logic applies independently (no license here).
    assert findings, "unlicensed M2-form sentence must still be a finding"


def test_no_warning_without_mechanism_pattern():
    warnings: list = []
    findings = scan_text(
        "The benchmark uses no latent user model.", warnings=warnings
    )
    assert warnings == []
    assert findings == []


def test_cli_emits_mechanism_warning(tmp_path, capsys):
    doc = tmp_path / "warn.md"
    doc.write_text(_WARNING_SENTENCE + "\n", encoding="utf-8")
    rc = main([str(doc)])
    out = capsys.readouterr().out
    assert rc == 1  # unlicensed M2-form sentence
    assert "[경고]" in out
    assert "재검토" in out


# --- CLI / gate integration ---


def test_cli_mechanism_fail_closed_without_gate_args(tmp_path, capsys):
    doc = tmp_path / "m2.md"
    doc.write_text(_M2_CANONICAL + "\n", encoding="utf-8")
    rc = main([str(doc)])
    out = capsys.readouterr().out
    assert rc == 1
    # Korean Track M guidance: rewrite as hypothesis (M1) or obtain the
    # scope markers + ladder_gate license.
    assert "가설형(M1)" in out
    assert "ladder_gate" in out


def _write_gate_evidence(tmp_path, caveat_marker=_CAVEAT):
    prereg = tmp_path / "prereg.json"
    prereg.write_text(
        json.dumps({"tieBreakCaveatMarker": caveat_marker}), encoding="utf-8"
    )
    report = tmp_path / "report.json"
    report.write_text("{}", encoding="utf-8")
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"entries": []}), encoding="utf-8")
    return prereg, report, ledger


def _gate_args(prereg, report, ledger):
    return [
        "--prereg", str(prereg),
        "--reports", str(report),
        "--ledger", str(ledger),
    ]


def test_cli_gate_grants_m2_license(tmp_path, capsys, monkeypatch):
    import echo_bench.tools.ladder_gate as lg

    monkeypatch.setattr(
        lg,
        "evaluate_mechanism_license",
        lambda *a, **k: {"rungs": {"M2": True}, "caveatRequired": False},
    )
    prereg, report, ledger = _write_gate_evidence(tmp_path)
    doc = tmp_path / "m2.md"
    doc.write_text(_M2_CANONICAL + "\n", encoding="utf-8")
    rc = main([str(doc)] + _gate_args(prereg, report, ledger))
    assert rc == 0, capsys.readouterr().out


def test_cli_gate_denies_m2_license(tmp_path, capsys, monkeypatch):
    import echo_bench.tools.ladder_gate as lg

    monkeypatch.setattr(
        lg,
        "evaluate_mechanism_license",
        lambda *a, **k: {"rungs": {"M2": False}, "caveatRequired": False},
    )
    prereg, report, ledger = _write_gate_evidence(tmp_path)
    doc = tmp_path / "m2.md"
    doc.write_text(_M2_CANONICAL + "\n", encoding="utf-8")
    rc = main([str(doc)] + _gate_args(prereg, report, ledger))
    assert rc == 1


def test_cli_gate_caveat_marker_read_from_prereg(tmp_path, capsys, monkeypatch):
    import echo_bench.tools.ladder_gate as lg

    monkeypatch.setattr(
        lg,
        "evaluate_mechanism_license",
        lambda *a, **k: {"rungs": {"M2": True}, "caveatRequired": True},
    )
    custom = "subject to a CUSTOM ordering caveat (TEST-001)"
    prereg, report, ledger = _write_gate_evidence(tmp_path, caveat_marker=custom)

    # Sentence carrying the prereg's custom marker passes...
    ok_doc = tmp_path / "ok.md"
    ok_doc.write_text(
        _M2_CANONICAL[:-1] + ", " + custom + ".\n", encoding="utf-8"
    )
    rc = main([str(ok_doc)] + _gate_args(prereg, report, ledger))
    assert rc == 0, capsys.readouterr().out

    # ...the default marker no longer satisfies the prereg's custom marker.
    bad_doc = tmp_path / "bad.md"
    bad_doc.write_text(_M2_WITH_CAVEAT + "\n", encoding="utf-8")
    rc = main([str(bad_doc)] + _gate_args(prereg, report, ledger))
    assert rc == 1


def test_cli_partial_gate_args_fail_closed(tmp_path, capsys):
    prereg, _, _ = _write_gate_evidence(tmp_path)
    doc = tmp_path / "m2.md"
    doc.write_text(_M2_CANONICAL + "\n", encoding="utf-8")
    rc = main([str(doc), "--prereg", str(prereg)])
    assert rc == 1  # missing --reports/--ledger -> no license, fail closed


# --- Regression: live tree stays clean with the Track M layer active ---


def test_live_tree_clean_with_mechanism_layer():
    warnings: list = []
    findings = scan_paths([_DOCS_DIR, _REPORTS_DIR], warnings=warnings)
    mech = [f for f in findings if f.phrase in MECHANISM_CLAIM_PATTERNS]
    assert mech == [], (
        "live docs/ + outputs/reports/ tripped a Track M mechanism pattern "
        f"(report as a concern, do not weaken the regex): {mech}"
    )
    assert warnings == [], f"live tree produced Track M warnings: {warnings}"

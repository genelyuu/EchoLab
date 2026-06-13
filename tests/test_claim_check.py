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


# ---------------------------------------------------------------------------
# G-022a review fixes — new fixtures
# ---------------------------------------------------------------------------

# Fix 1: backtick-straddle over-suppression
_STRADDLE_SENTENCE = (
    "The `axs_ucb_default` arm shows that amplification requires "
    "history-dependent exploration in the `replay` audit."
)
_GENUINE_BACKTICK_SENTENCE = (
    "The form `requires history-dependent exploration` is a quarantined "
    "mechanism claim form."
)


def test_backtick_straddle_is_flagged():
    """A mechanism claim flanked by UNRELATED backtick spans must be a finding.

    The pattern is: `identifier` ... CLAIM ... `identifier` — the match is NOT
    inside any open backtick span; the sentence-level check must NOT suppress it.
    """
    findings = scan_text(_STRADDLE_SENTENCE)
    assert findings, (
        "backtick-straddle sentence wrongly suppressed — "
        "unrelated code spans must not protect a prose claim: "
        f"{_STRADDLE_SENTENCE!r}"
    )


def test_genuine_backtick_span_still_suppressed():
    """A claim fully enclosed in a single backtick span must remain suppressed."""
    findings = scan_text(_GENUINE_BACKTICK_SENTENCE)
    assert findings == [], (
        f"genuine backtick-enclosed mention wrongly flagged: {_GENUINE_BACKTICK_SENTENCE!r}"
    )


# Fix 2: regression fixtures for multi-line wrap and marker smuggling

_M2_MULTILINE = (
    "Within the tested policy families\n"
    "in this controlled testbed, above-null slate separability requires\n"
    "history-dependent exploration pressure."
)


def test_multiline_m2_unlicensed_one_finding():
    """M2 sentence split across 3 lines, unlicensed → exactly one finding."""
    findings = scan_text(_M2_MULTILINE)
    assert len(findings) == 1, (
        f"expected exactly one finding for multi-line M2, got: {findings}"
    )


def test_multiline_m2_licensed_clean():
    """Same sentence with an active license (no caveat required) → clean."""
    findings = scan_text(
        _M2_MULTILINE,
        mechanism_license={"m2": True, "caveatRequired": False},
    )
    assert findings == [], f"licensed multi-line M2 wrongly flagged: {findings}"


_SMUGGLE_TEXT = (
    "We hypothesize nothing here.\n"
    "Amplification requires history-dependent exploration."
)


def test_marker_smuggling_second_sentence_flagged():
    """Hypothesis marker in sentence 1 must NOT license the assertion in sentence 2."""
    findings = scan_text(_SMUGGLE_TEXT)
    assert findings, (
        "hypothesis marker in prior sentence must not license the assertion: "
        f"{_SMUGGLE_TEXT!r}"
    )


# Fix 3: unicode evasion normalization

def test_unicode_hyphen_evasion_flagged():
    """U+2011 non-breaking hyphen in 'history-dependent' must not bypass the pattern."""
    # U+2011 is a non-breaking hyphen — visually identical to ASCII '-'
    evasion = "Amplification requires history‑dependent exploration."
    findings = scan_text(evasion)
    assert findings, (
        f"U+2011 hyphen evasion wrongly suppressed: {evasion!r}"
    )


def test_zwsp_evasion_flagged():
    """Zero-width space (U+200B) inserted into 'requires' must not bypass the pattern."""
    evasion = "Amplification requ​ires history-dependent exploration."
    findings = scan_text(evasion)
    assert findings, (
        f"ZWSP evasion wrongly suppressed: {evasion!r}"
    )


# Fix 4: KeyError traceback in gate wiring

def test_gate_malformed_prereg_no_traceback(tmp_path, capsys, monkeypatch):
    """A malformed prereg missing 'preregId' must not raise a traceback.

    The gate wiring must catch KeyError/TypeError, emit the Korean warning, and
    return exit 1 (or 0 for a clean file) — but never let an unhandled exception
    surface.
    """
    import echo_bench.tools.ladder_gate as lg

    def _raise_key_error(*a, **kw):
        d = {}
        return d["preregId"]  # KeyError

    monkeypatch.setattr(lg, "evaluate_mechanism_license", _raise_key_error)
    prereg, report, ledger = _write_gate_evidence(tmp_path)

    # File with an M2 sentence (unlicensed because gate raises) → exit 1
    doc = tmp_path / "m2.md"
    doc.write_text(_M2_CANONICAL + "\n", encoding="utf-8")

    # Should not raise; captured output must contain the Korean warning
    rc = main([str(doc)] + _gate_args(prereg, report, ledger))
    out = capsys.readouterr().out
    assert rc == 1, f"expected exit 1 (unlicensed), got {rc}"
    assert "[경고]" in out, f"Korean warning expected in output: {out!r}"


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


# ---------------------------------------------------------------------------
# G-022a-v3 (N2): Track M v3 "imprint-washout" vocabulary patterns.
# ---------------------------------------------------------------------------

from echo_bench.tools.claim_check import (  # noqa: E402
    MECHANISM_CLAIM_PATTERNS_V3,
)

# Canonical v3 M2 sentence from axs_mechanism_prereg_v3_draft.json
# branches[0].licensedClaim:
_V3_M2_CANONICAL = (
    "Within the tested policy families in this controlled testbed, above-null "
    "slate separability is amplified when trace-conditioned policy state is "
    "frozen early in the horizon and is eliminated by trace-independent bonus "
    "randomization; continual trace-conditioned updates attenuate but do not "
    "eliminate it."
)

# Same sentence with caveat marker spliced in before the final period.
_V3_M2_WITH_CAVEAT = _V3_M2_CANONICAL[:-1] + ", " + _CAVEAT + "."

# Paraphrased (wrong) caveat.
_V3_M2_PARAPHRASED_CAVEAT = (
    _V3_M2_CANONICAL[:-1] + ", subject to minor ordering sensitivity."
)

# ---------------------------------------------------------------------------
# v3 pattern constants (for assert-on-pattern tests)
# ---------------------------------------------------------------------------
# N2 review fixes reflected here:
#   Fix 1: frozen\s+.{0,40}state → frozen\s+[^.\n]{0,40}state in both amplify patterns
#   Fix 2: entry[-\s]point removed from _PAT_V2_ENTRY / _PAT_V2_ENTRY_REV;
#           new _PAT_V2_ENTRY_POINT / _PAT_V2_ENTRY_POINT_REV require mechanism anchor
#   Fix 3a: washout split out of _PAT_V3_IMPRINT_CAUSAL / _PAT_V3_IMPRINT_CAUSAL_REV;
#            new _PAT_V3_WASHOUT_CAUSAL / _PAT_V3_WASHOUT_CAUSAL_REV require anchor
#   Fix 3b: freez(?:e|ing) split out of _PAT_V3_AMPLIFY_REV; new
#            _PAT_V3_AMPLIFY_FREEZE_ANCHOR requires mechanism anchor in sentence

_PAT_V3_AMPLIFY = (
    r"amplif(?:y|ies|ied)[^.\n]{0,80}"
    r"(?:trace[-\s]conditioned|frozen\s+[^.\n]{0,40}state|freez(?:e|ing)|imprint(?:ing)?)"
)
# noun-first: inherently mechanism-specific nouns only (bare freeze removed — Fix 3b)
_PAT_V3_AMPLIFY_REV = (
    r"(?:trace[-\s]conditioned|frozen\s+[^.\n]{0,40}state|imprint(?:ing)?)"
    r"[^.\n]{0,80}amplif(?:y|ies|ied)"
)
# noun-first bare freeze with mechanism-context anchor (Fix 3b)
_PAT_V3_AMPLIFY_FREEZE_ANCHOR = (
    r"(?:trace|polic|state|imprint|separabilit|probe|slate)[^.\n]{0,160}freez(?:e|ing)[^.\n]{0,80}amplif(?:y|ies|ied)"
    r"|freez(?:e|ing)[^.\n]{0,80}(?:trace|polic|state|imprint|separabilit|probe|slate)[^.\n]{0,80}amplif(?:y|ies|ied)"
    r"|freez(?:e|ing)[^.\n]{0,80}amplif(?:y|ies|ied)[^.\n]{0,80}(?:trace|polic|state|imprint|separabilit|probe|slate)"
)
_PAT_V3_ELIM = (
    r"eliminat(?:e|es|ed|ion)[^.\n]{0,80}"
    r"trace[-\s]independent[^.\n]{0,40}(?:bonus|score|randomi[sz]ation)"
)
_PAT_V3_ELIM_REV = (
    r"trace[-\s]independent[^.\n]{0,40}(?:bonus|score|randomi[sz]ation)"
    r"[^.\n]{0,80}eliminat(?:e|es|ed|ion)"
)
_PAT_V3_ATTEN = (
    r"attenuat(?:e|es|ed)[^.\n]{0,80}"
    r"(?:separability|imprint|trace[-\s]conditioned\s+updates?)"
)
_PAT_V3_ATTEN_REV = (
    r"(?:separability|imprint|trace[-\s]conditioned\s+updates?)"
    r"[^.\n]{0,80}attenuat(?:e|es|ed)"
)
# imprint-only causal patterns (washout removed — Fix 3a)
_PAT_V3_IMPRINT_CAUSAL = (
    r"(?:requires|driven\s+by|attributable\s+to|caus(?:ed|es|ing))"
    r"[^.\n]{0,80}imprint(?:ing)?"
)
_PAT_V3_IMPRINT_CAUSAL_REV = (
    r"imprint(?:ing)?"
    r"[^.\n]{0,80}(?:requires|driven\s+by|attributable\s+to|caus(?:ed|es|ing))"
)
# washout causal patterns with mechanism-context anchor (Fix 3a)
_PAT_V3_WASHOUT_CAUSAL = (
    r"(?:separabilit|probe|slate|trace|polic|state|imprint)[^.\n]{0,160}(?:requires|driven\s+by|attributable\s+to|caus(?:ed|es|ing))[^.\n]{0,80}washout"
    r"|(?:requires|driven\s+by|attributable\s+to|caus(?:ed|es|ing))[^.\n]{0,80}washout[^.\n]{0,80}(?:separabilit|probe|slate|trace|polic|state|imprint)"
)
_PAT_V3_WASHOUT_CAUSAL_REV = (
    r"(?:separabilit|probe|slate|trace|polic|state|imprint)[^.\n]{0,80}washout[^.\n]{0,80}(?:requires|driven\s+by|attributable\s+to|caus(?:ed|es|ing))"
    r"|washout[^.\n]{0,80}(?:requires|driven\s+by|attributable\s+to|caus(?:ed|es|ing))[^.\n]{0,80}(?:separabilit|probe|slate|trace|polic|state|imprint)"
)
# v2-era entry noun group without entry[-\s]point (Fix 2)
_PAT_V2_ENTRY = (
    r"(?:requires|driven\s+by|attributable\s+to|caus(?:ed|es|ing))"
    r"[^.\n]{0,80}"
    r"(?:context[-\s]feature\s+(?:path|channel|pathway|entry)"
    r"|mean[-\s]value\s+path"
    r"|learned[-\s]weight\s+(?:path|channel)|selection\s+pathway)"
)
_PAT_V2_ENTRY_REV = (
    r"(?:context[-\s]feature\s+(?:path|channel|pathway|entry)"
    r"|mean[-\s]value\s+path"
    r"|learned[-\s]weight\s+(?:path|channel)|selection\s+pathway)"
    r"[^.\n]{0,80}(?:requires|driven\s+by|attributable\s+to|caus(?:ed|es|ing))"
)
# entry-point patterns with mechanism-context anchor (Fix 2)
_PAT_V2_ENTRY_POINT = (
    r"(?:separabilit|probe|slate|mechanism)[^.\n]{0,160}(?:requires|driven\s+by|attributable\s+to|caus(?:ed|es|ing))[^.\n]{0,80}entry[-\s]point"
    r"|(?:requires|driven\s+by|attributable\s+to|caus(?:ed|es|ing))[^.\n]{0,80}entry[-\s]point[^.\n]{0,80}(?:separabilit|probe|slate|mechanism)"
)
_PAT_V2_ENTRY_POINT_REV = (
    r"(?:separabilit|probe|slate|mechanism)[^.\n]{0,80}entry[-\s]point[^.\n]{0,80}(?:requires|driven\s+by|attributable\s+to|caus(?:ed|es|ing))"
    r"|entry[-\s]point[^.\n]{0,80}(?:requires|driven\s+by|attributable\s+to|caus(?:ed|es|ing))[^.\n]{0,80}(?:separabilit|probe|slate|mechanism)"
)


def test_v3_pattern_tuple_exists():
    """MECHANISM_CLAIM_PATTERNS_V3 must be a non-empty tuple of strings."""
    assert isinstance(MECHANISM_CLAIM_PATTERNS_V3, tuple)
    assert len(MECHANISM_CLAIM_PATTERNS_V3) > 0
    for p in MECHANISM_CLAIM_PATTERNS_V3:
        assert isinstance(p, str), f"pattern must be str: {p!r}"


def test_v3_patterns_contain_required_patterns():
    """v3 pattern tuple must contain all required vocabulary patterns."""
    for required in (
        _PAT_V3_AMPLIFY,
        _PAT_V3_AMPLIFY_REV,
        _PAT_V3_AMPLIFY_FREEZE_ANCHOR,
        _PAT_V3_ELIM,
        _PAT_V3_ELIM_REV,
        _PAT_V3_ATTEN,
        _PAT_V3_ATTEN_REV,
        _PAT_V3_IMPRINT_CAUSAL,
        _PAT_V3_IMPRINT_CAUSAL_REV,
        _PAT_V3_WASHOUT_CAUSAL,
        _PAT_V3_WASHOUT_CAUSAL_REV,
        _PAT_V2_ENTRY,
        _PAT_V2_ENTRY_REV,
        _PAT_V2_ENTRY_POINT,
        _PAT_V2_ENTRY_POINT_REV,
    ):
        assert required in MECHANISM_CLAIM_PATTERNS_V3, (
            f"v3 pattern missing from MECHANISM_CLAIM_PATTERNS_V3: {required!r}"
        )


def test_v1_patterns_byte_identical():
    """MECHANISM_CLAIM_PATTERNS (v1) must not be modified — byte-identical."""
    assert MECHANISM_CLAIM_PATTERNS == (
        _PAT_REQUIRES,
        _PAT_ATTRIBUTABLE,
        _PAT_DRIVEN,
        _PAT_CAUSES,
        _PAT_NOT_ADAPTIVITY,
    )


# ---------------------------------------------------------------------------
# FAIL fixtures (no marker/license → finding)
# ---------------------------------------------------------------------------


def test_v3_canonical_m2_unlicensed_flagged():
    """Canonical v3 M2 sentence as plain prose must be a finding (no license)."""
    findings = scan_text(_V3_M2_CANONICAL)
    assert findings, (
        f"v3 M2 canonical sentence must be flagged without license: {_V3_M2_CANONICAL!r}"
    )
    # The phrase on each finding must be a v3 pattern.
    for f in findings:
        assert f.phrase in MECHANISM_CLAIM_PATTERNS_V3, (
            f"finding phrase {f.phrase!r} not in MECHANISM_CLAIM_PATTERNS_V3"
        )


def test_v3_amplify_fragment_flagged():
    """Amplification + trace-conditioned fragment must be flagged."""
    text = (
        "Above-null slate separability is amplified when "
        "trace-conditioned policy state is frozen early in the horizon."
    )
    findings = scan_text(text)
    assert findings, f"amplify+trace-conditioned fragment must be flagged: {text!r}"


def test_v3_elimination_flagged():
    """Elimination + trace-independent bonus randomization must be flagged."""
    text = "Separability is eliminated by trace-independent bonus randomization."
    findings = scan_text(text)
    assert findings, f"elimination pattern must be flagged: {text!r}"


def test_v3_imprint_causal_verb_flagged():
    """Slate separability driven by imprinting must be flagged."""
    text = "Slate separability is driven by early imprinting of the learned state."
    findings = scan_text(text)
    assert findings, f"imprint/washout causal pattern must be flagged: {text!r}"


def test_v2_era_context_feature_pathway_flagged():
    """v2-era hole: context-feature pathway as mechanism must be flagged."""
    text = "Probe separability is attributable to the context-feature pathway."
    findings = scan_text(text)
    assert findings, f"v2-era context-feature pathway pattern must be flagged: {text!r}"


def test_v1_m2_canonical_still_flagged():
    """Regression: v1 M2 canonical sentence must still be caught by v1 patterns."""
    findings = scan_text(_M2_CANONICAL)
    assert findings, f"v1 M2 canonical sentence must still be flagged: {_M2_CANONICAL!r}"
    v1_phrases = {f.phrase for f in findings}
    assert any(p in MECHANISM_CLAIM_PATTERNS for p in v1_phrases), (
        f"v1 canonical must trip at least one v1 pattern; got: {v1_phrases}"
    )


# ---------------------------------------------------------------------------
# PASS fixtures (allowed: license, hypothesis marker, technical prose, code span)
# ---------------------------------------------------------------------------


def test_v3_canonical_m2_licensed_no_caveat_passes():
    """v3 M2 sentence with active M2 license (caveatRequired=False) → clean."""
    findings = scan_text(
        _V3_M2_CANONICAL,
        mechanism_license={"m2": True, "caveatRequired": False},
    )
    assert findings == [], f"licensed v3 M2 sentence (no caveat) wrongly flagged: {findings}"


def test_v3_canonical_m2_licensed_with_caveat_passes():
    """v3 M2 sentence + M2 license + canonical caveat marker → clean."""
    findings = scan_text(
        _V3_M2_WITH_CAVEAT,
        mechanism_license={"m2": True, "caveatRequired": True, "caveatMarker": _CAVEAT},
    )
    assert findings == [], (
        f"licensed v3 M2 sentence with caveat wrongly flagged: {findings}"
    )


def test_v3_canonical_m2_licensed_caveat_required_absent_flagged():
    """v3 M2 sentence + license + caveatRequired=True but caveat absent → finding."""
    findings = scan_text(
        _V3_M2_CANONICAL,
        mechanism_license={"m2": True, "caveatRequired": True, "caveatMarker": _CAVEAT},
    )
    assert findings, (
        "caveatRequired=True without the caveat in v3 M2 sentence must be a finding"
    )


def test_v3_canonical_m2_paraphrased_caveat_flagged():
    """v3 M2 sentence + license + paraphrased caveat → finding."""
    findings = scan_text(
        _V3_M2_PARAPHRASED_CAVEAT,
        mechanism_license={"m2": True, "caveatRequired": True, "caveatMarker": _CAVEAT},
    )
    assert findings, "paraphrased caveat must NOT satisfy the v3 M2 caveat rule"


def test_v3_m1_early_freezing_hypothesis_passes():
    """M1 hypothesis marker with v3 vocabulary → clean."""
    text = (
        "We hypothesize that early freezing of trace-conditioned state "
        "may amplify probe separability."
    )
    findings = scan_text(text)
    assert findings == [], f"M1 hypothesis form (v3 vocab) wrongly flagged: {text!r}"


def test_v3_m1_consistent_with_imprint_washout_passes():
    """'is consistent with' is an M1 marker → v3 vocabulary allowed."""
    text = "Pilot evidence is consistent with an imprint-washout account."
    findings = scan_text(text)
    assert findings == [], f"'is consistent with' M1 form wrongly flagged: {text!r}"


def test_v3_technical_freeze_round_passes():
    """Technical descriptor prose with 'freeze' but no mechanism coupling → clean."""
    text = "The freeze_round seam truncates the bandit replay view."
    findings = scan_text(text)
    assert findings == [], f"technical freeze_round descriptor wrongly flagged: {text!r}"


def test_v3_renderer_eliminates_degenerate_rasters_passes():
    """'eliminates' with a non-mechanism noun (rasters) must not fire."""
    text = "The renderer eliminates degenerate rasters before pooling."
    findings = scan_text(text)
    assert findings == [], (
        f"'eliminates degenerate rasters' (non-mechanism noun) wrongly flagged: {text!r}"
    )


def test_v3_amplifier_gain_identifier_passes():
    """'amplifier gain parameter' style prose must not fire the amplify pattern."""
    text = "Set the amplifier gain parameter to 1.0 before the sweep."
    findings = scan_text(text)
    assert findings == [], f"amplifier gain parameter identifier wrongly flagged: {text!r}"


def test_v3_code_span_mention_passes():
    """Backtick code-span mention of the v3 canonical sentence → clean (mention, not assertion)."""
    text = (
        "The registered claim form is: `above-null slate separability is amplified "
        "when trace-conditioned policy state is frozen early in the horizon and is "
        "eliminated by trace-independent bonus randomization`."
    )
    findings = scan_text(text)
    assert findings == [], f"code-span mention of v3 sentence wrongly flagged: {text!r}"


def test_v3_user_vocab_warning_triggered():
    """v3-pattern sentence + 'users' co-occurrence triggers the WARNING path."""
    text = (
        "Within the tested policy families in this controlled testbed, "
        "separability is amplified when trace-conditioned state is frozen "
        "and users observe the effect."
    )
    warnings: list = []
    scan_text(text, warnings=warnings)
    assert any("[경고]" in w for w in warnings), (
        f"mechanism+user co-occurrence must trigger Korean WARNING: {warnings!r}"
    )


# ---------------------------------------------------------------------------
# N2 review fix fixtures
# ---------------------------------------------------------------------------

# Fix 1: Gap-spec — frozen\s+[^.\n]{0,40}state (not .)

def test_fix1_frozen_state_no_newline_crossing():
    r"""frozen\s+[^.\n]{0,40}state must not cross a sentence boundary (newline)."""
    import re as _re
    verb_first = _PAT_V3_AMPLIFY
    noun_first = _PAT_V3_AMPLIFY_REV
    cross_newline = "frozen policy\nstate"
    assert _re.search(r"frozen\s+[^.\n]{0,40}state", cross_newline) is None, (
        "frozen..state pattern must not cross a newline — gap-spec violation"
    )
    # Sanity: the verb-first and noun-first patterns reference [^.\n]{0,40}
    assert r"[^.\n]{0,40}state" in verb_first, (
        "verb-first amplify pattern must use [^.\\n]{0,40}state (Fix 1)"
    )
    assert r"[^.\n]{0,40}state" in noun_first, (
        "noun-first amplify pattern must use [^.\\n]{0,40}state (Fix 1)"
    )


# Fix 2: entry-point over-suppression

def test_fix2_cli_entry_point_verb_first_passes():
    """'The CLI entry-point requires three positional arguments.' must PASS (no mechanism anchor)."""
    assert scan_text("The CLI entry-point requires three positional arguments.") == [], (
        "plain CLI entry-point prose must not be flagged"
    )


def test_fix2_cli_entry_point_noun_first_passes():
    """'The entry-point requires a config path.' must PASS (no mechanism anchor)."""
    assert scan_text("The entry-point requires a config path.") == [], (
        "plain CLI noun-first entry-point prose must not be flagged"
    )


def test_fix2_cli_entry_point_multiple_args_passes():
    """'This entry point requires a seed and a policy name.' must PASS."""
    assert scan_text("This entry point requires a seed and a policy name.") == [], (
        "generic entry-point prose must not be flagged"
    )


def test_fix2_entry_point_with_separability_fails():
    """'Above-null slate separability requires the context entry-point.' must FAIL."""
    findings = scan_text("Above-null slate separability requires the context entry-point.")
    assert findings, (
        "entry-point sentence with separability/slate anchor must be flagged"
    )


# Fix 3a: washout bare coupling

def test_fix3a_solver_washout_passes():
    """'The solver requires a washout period of 50 steps before sampling.' must PASS."""
    assert scan_text(
        "The solver requires a washout period of 50 steps before sampling."
    ) == [], (
        "plain solver washout sentence (no mechanism anchor) must not be flagged"
    )


def test_fix3a_slate_separability_washout_fails():
    """'Slate separability requires update washout.' must FAIL (mechanism anchor present)."""
    findings = scan_text("Slate separability requires update washout.")
    assert findings, (
        "washout sentence with slate/separability anchor must be flagged"
    )


def test_fix3a_imprint_causal_still_flagged():
    """imprint-causal catch fixture must still fire (regression check)."""
    text = "Slate separability is driven by early imprinting of the learned state."
    findings = scan_text(text)
    assert findings, f"imprint causal pattern must still be flagged after Fix 3a: {text!r}"
    assert any(f.phrase == _PAT_V3_IMPRINT_CAUSAL for f in findings), (
        f"must fire on imprint causal pattern; got: {[f.phrase for f in findings]}"
    )


# Fix 3b: noun-first bare freeze coupling

def test_fix3b_colormap_freeze_amplify_passes():
    """'Freezing the colormap amplifies banding artifacts in the preview.' must PASS."""
    assert scan_text(
        "Freezing the colormap amplifies banding artifacts in the preview."
    ) == [], (
        "bare freeze/amplify with no mechanism anchor must not be flagged"
    )


def test_fix3b_early_freezing_trace_conditioned_fails():
    """'Early freezing of trace-conditioned state amplifies separability.' must FAIL."""
    findings = scan_text(
        "Early freezing of trace-conditioned state amplifies separability."
    )
    assert findings, (
        "freeze+trace-conditioned noun-first amplify must be flagged"
    )


def test_fix3b_frozen_policy_state_amplify_fails():
    """'Frozen policy state amplifies probe separability.' must FAIL."""
    findings = scan_text("Frozen policy state amplifies probe separability.")
    assert findings, (
        "frozen policy state amplifies probe separability must be flagged"
    )


def test_fix3b_amplify_pattern_verb_first_still_catches_freeze():
    """Verb-first amplify with freez still fires — no anchor required for verb-first."""
    # verb-first: amplif ... freez — the amplif verb is the mechanism anchor
    text = "The update amplifies freezing artifacts in the probe output."
    findings = scan_text(text)
    assert findings, (
        "verb-first amplif...freez must still be caught after Fix 3b"
    )


# ---------------------------------------------------------------------------
# N7-4 (claim scanner v3.1): alpha/bonus suppression coupling + per-track
# M2 sentence licensing.
# ---------------------------------------------------------------------------

from echo_bench.tools.claim_check import (  # noqa: E402
    MECHANISM_CLAIM_PATTERNS_V3_1,
)

# Canonical per-track sentences are pulled FROM the committed prereg v3 draft
# (data-derived — no hardcoded sentence literals).
_PREREG_V3_DRAFT_PATH = (
    _REPO_ROOT / "configs" / "prereg" / "axs_mechanism_prereg_v3_draft.json"
)
_PREREG_V3_DRAFT = json.loads(_PREREG_V3_DRAFT_PATH.read_text(encoding="utf-8"))
_CANONICAL_SENTENCES = _PREREG_V3_DRAFT["canonicalSentences"]
_M_IMP = _CANONICAL_SENTENCES["M-IMP"]
_M_NOISE = _CANONICAL_SENTENCES["M-NOISE"]

# Scope-marker wrapper: since draftRevision 3 the per-track canonical
# sentences themselves carry BOTH registered scope markers, so the wrapper is
# no longer required for publication — it remains here to pin that an embedded
# verbatim canonical inside a larger scoped sentence is ALSO licensed (the
# substring rule), independent of the bare-canonical path.
_SCOPE_PREFIX = (
    "Within the tested policy families in this controlled testbed, "
    "the gate-licensed claim is recorded verbatim: "
)
_M_IMP_SCOPED = _SCOPE_PREFIX + _M_IMP
_M_NOISE_SCOPED = _SCOPE_PREFIX + _M_NOISE

# Same with the canonical caveat marker spliced in BEFORE the embedded
# canonical sentence (the canonical's own final period ends the sentence, so
# the caveat cannot follow it within the same sentence).
_M_IMP_SCOPED_CAVEAT = (
    "Within the tested policy families in this controlled testbed, "
    + _CAVEAT
    + ", the gate-licensed claim is recorded verbatim: "
    + _M_IMP
)

# v3.1 pattern constants (pin the exact strings, mirroring the V3 convention).
_SEP_NOUNS = (
    r"(?:separabilit(?:y|ies)|probe[-\s]separability|slate[-\s]separability"
    r"|trajectory\s+distinction)"
)
_PAT_V31_AGENT_FIRST = (
    r"(?:alpha|exploration\s+bonus|bonus)[^.\n]{0,80}"
    r"(?:suppress(?:es|ed)?|reduc(?:es|ed)|attenuat(?:es|ed))[^.\n]{0,80}"
    + _SEP_NOUNS
)
_PAT_V31_NOUN_PASSIVE = (
    _SEP_NOUNS
    + r"[^.\n]{0,80}(?:is|are|was|were)\s+(?:suppressed|reduced|attenuated)\s+by"
    r"[^.\n]{0,80}(?:alpha|exploration\s+bonus|bonus)"
)
_PAT_V31_NOUN_ACTIVE = (
    _SEP_NOUNS
    + r"[^.\n]{0,80}(?:suppress(?:es|ed)?|reduc(?:es|ed)|attenuat(?:es|ed))"
    r"[^.\n]{0,80}(?:alpha|exploration\s+bonus|bonus)"
)
_PAT_V31_DISRUPT = r"disrupt(?:s|ed|ing)?[^.\n]{0,80}" + _SEP_NOUNS
_PAT_V31_DISRUPT_REV = _SEP_NOUNS + r"[^.\n]{0,80}disrupt(?:s|ed|ing)?"


def test_v31_pattern_tuple_exists_and_pins_patterns():
    """MECHANISM_CLAIM_PATTERNS_V3_1 must exist and contain the new patterns."""
    assert isinstance(MECHANISM_CLAIM_PATTERNS_V3_1, tuple)
    for required in (
        _PAT_V31_AGENT_FIRST,
        _PAT_V31_NOUN_PASSIVE,
        _PAT_V31_NOUN_ACTIVE,
        _PAT_V31_DISRUPT,
        _PAT_V31_DISRUPT_REV,
    ):
        assert required in MECHANISM_CLAIM_PATTERNS_V3_1, (
            f"v3.1 pattern missing from MECHANISM_CLAIM_PATTERNS_V3_1: {required!r}"
        )
    for p in MECHANISM_CLAIM_PATTERNS_V3_1:
        assert isinstance(p, str)
        assert ".{0," not in p.replace("[^.\\n]{0,", ""), (
            f"unbounded-class gap forbidden in v3.1 pattern: {p!r}"
        )


def test_v31_does_not_touch_pinned_tuples():
    """v1 and V3 tuples remain byte-identical (re-assert the pins)."""
    assert MECHANISM_CLAIM_PATTERNS == (
        _PAT_REQUIRES,
        _PAT_ATTRIBUTABLE,
        _PAT_DRIVEN,
        _PAT_CAUSES,
        _PAT_NOT_ADAPTIVITY,
    )
    assert len(MECHANISM_CLAIM_PATTERNS_V3) == 15
    # No overlap: v3.1 patterns are NEW, not relabeled v1/V3 patterns.
    assert not (
        set(MECHANISM_CLAIM_PATTERNS_V3_1)
        & (set(MECHANISM_CLAIM_PATTERNS) | set(MECHANISM_CLAIM_PATTERNS_V3))
    )


# --- Part 1 mandatory fixtures ---


def test_v31_alpha_suppresses_separability_flagged():
    """MEASURED GAP: 'Alpha suppresses separability.' must now be a finding."""
    findings = scan_text("Alpha suppresses separability.")
    assert findings, "'Alpha suppresses separability.' must be flagged"
    assert all(f.phrase in MECHANISM_CLAIM_PATTERNS_V3_1 for f in findings), (
        f"expected only v3.1 patterns to fire: {[f.phrase for f in findings]}"
    )


def test_v31_heterogeneous_alpha_observation_passes():
    """No suppression-verb/separability coupling → pass."""
    text = "We observed heterogeneous alpha effects in critic pilots."
    assert scan_text(text) == [], f"observation sentence wrongly flagged: {text!r}"


def test_v31_alpha_hypothesis_licensed_as_m1():
    """Hypothesis marker licenses the alpha-suppression coupling as M1.

    The regex itself MUST match (so the pass is the M1 path, not a pattern
    hole): 'hypothesize' is in MECHANISM_HYPOTHESIS_MARKERS.
    """
    import re as _re

    from echo_bench.tools.claim_check import MECHANISM_HYPOTHESIS_MARKERS

    text = "We hypothesize that alpha may suppress separability."
    assert any(
        _re.search(p, text, _re.IGNORECASE)
        for p in MECHANISM_CLAIM_PATTERNS_V3_1
    ), "the v3.1 coupling regex must match the hypothesis sentence"
    assert "hypothesize" in MECHANISM_HYPOTHESIS_MARKERS
    assert scan_text(text) == [], f"M1 hypothesis form wrongly flagged: {text!r}"


def test_v31_exploration_bonus_agent_flagged():
    findings = scan_text(
        "The exploration bonus suppresses probe separability in long horizons."
    )
    assert findings, "exploration-bonus agent coupling must be flagged"


def test_v31_noun_first_passive_flagged():
    findings = scan_text(
        "Probe separability was reduced by the exploration bonus."
    )
    assert findings, "noun-first passive (suppressed/reduced by agent) must be flagged"


def test_v31_noun_first_active_flagged():
    findings = scan_text(
        "Separability dropped when the schedule reduced the bonus contribution."
    )
    assert findings, "noun-first active order (noun..verb..agent) must be flagged"


@pytest.mark.parametrize(
    "text",
    [
        # No separability noun → plain technical prose.
        "The alpha parameter reduces the exploration bonus magnitude.",
        "We reduced the bonus schedule length.",
        # Identifier / backtick contexts.
        "Set `alpha=0.0` to disable the bonus term in the config.",
        "alpha=0.0 disables the bonus term",
        # Agent without any suppression verb.
        "The bonus term is added to the per-card score before ranking.",
    ],
)
def test_v31_plain_technical_prose_passes(text):
    assert scan_text(text) == [], f"technical prose wrongly flagged: {text!r}"


def test_v31_backtick_straddle_in_is_flagged():
    """Straddle-IN is DELIBERATE anti-evasion (controller-pinned, N7-4 review).

    A match that STARTS in prose and whose bounded ``[^.\\n]{0,N}`` gap walks
    INTO a backtick span IS a finding: only matches that start inside an open
    backtick span are suppressed. Otherwise wrapping just the noun in backticks
    ('alpha suppresses `separability`.') becomes an evasion channel.
    """
    evasion = "alpha suppresses `separability`."
    findings = scan_text(evasion)
    assert findings, (
        f"backtick-noun evasion attempt must remain a finding: {evasion!r}"
    )
    assert any(
        f.phrase in MECHANISM_CLAIM_PATTERNS_V3_1 for f in findings
    ), f"expected a v3.1 coupling pattern to fire: {[f.phrase for f in findings]}"


def test_v31_alpha_attenuates_separability_fires_both_layers():
    """attenuate↔separability already fires V3 (agent-free); v3.1 adds the
    agent-coupled finding — the sentence honestly yields findings from BOTH
    tuples (one Finding per matching pattern; no dedup across tuples)."""
    findings = scan_text("Alpha attenuates probe separability.")
    phrases = {f.phrase for f in findings}
    assert any(p in MECHANISM_CLAIM_PATTERNS_V3 for p in phrases), (
        f"V3 attenuation coupling must still fire: {phrases}"
    )
    assert any(p in MECHANISM_CLAIM_PATTERNS_V3_1 for p in phrases), (
        f"v3.1 agent-coupled attenuation must also fire: {phrases}"
    )


def test_v31_regression_v1_and_v3_still_caught():
    # v1 exploration vocabulary still caught.
    assert scan_text(
        "Amplification generally requires history-dependent exploration."
    ), "v1 exploration-vocabulary sentence must still be caught"
    # V3 imprint/washout vocabulary still caught.
    assert scan_text(
        "Slate separability is driven by early imprinting of the learned state."
    ), "V3 imprint sentence must still be caught"
    assert scan_text(
        "Slate separability requires update washout."
    ), "V3 washout sentence must still be caught"


def test_v31_cli_alpha_suppression_korean_guidance(tmp_path, capsys):
    """CLI: a v3.1 hit exits 1 and prints the Track M Korean guidance."""
    doc = tmp_path / "alpha.md"
    doc.write_text("Alpha suppresses separability.\n", encoding="utf-8")
    rc = main([str(doc)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "[Track M]" in out
    assert "가설형(M1)" in out


# --- Per-track canonical-sentence visibility ---


def test_v31_m_noise_canonical_unlicensed_flagged():
    """The M-NOISE canonical sentence must be scanner-visible: licensing it is
    meaningless if the unlicensed sentence silently passes."""
    findings = scan_text(_M_NOISE)
    assert findings, (
        f"M-NOISE canonical sentence must be flagged without license: {_M_NOISE!r}"
    )


def test_v31_m_imp_canonical_unlicensed_flagged():
    findings = scan_text(_M_IMP)
    assert findings, (
        f"M-IMP canonical sentence must be flagged without license: {_M_IMP!r}"
    )


# --- Part 2: per-track M2 sentence licensing (gate-team contract) ---


def _v3_gate_result(licensed, branch, caveat_required=False):
    """Synthetic v3-shaped ladder_gate result per the gate-team contract."""
    return {
        "rungs": {
            "M0": True,
            "M1": True,
            "M2-IMP": _M_IMP in licensed,
            "M2-NOISE": _M_NOISE in licensed,
            "M2": (_M_IMP in licensed) and (_M_NOISE in licensed),
            "M3": False,
        },
        "branch": branch,
        "caveatRequired": caveat_required,
        "licensedSentences": list(licensed),
    }


def test_v31_per_track_imp_only_licensed(tmp_path, capsys, monkeypatch):
    """(a) M2-IMP only: M-IMP scoped sentence passes; M-NOISE is a finding."""
    import echo_bench.tools.ladder_gate as lg

    monkeypatch.setattr(
        lg,
        "evaluate_mechanism_license",
        lambda *a, **k: _v3_gate_result([_M_IMP], "imprint_only_supported"),
    )
    prereg, report, ledger = _write_gate_evidence(tmp_path)

    ok_doc = tmp_path / "imp.md"
    ok_doc.write_text(_M_IMP_SCOPED + "\n", encoding="utf-8")
    rc = main([str(ok_doc)] + _gate_args(prereg, report, ledger))
    assert rc == 0, capsys.readouterr().out

    bad_doc = tmp_path / "noise.md"
    bad_doc.write_text(_M_NOISE_SCOPED + "\n", encoding="utf-8")
    rc = main([str(bad_doc)] + _gate_args(prereg, report, ledger))
    assert rc == 1, "M-NOISE sentence must be a finding when only M2-IMP is licensed"


def test_v31_per_track_both_licensed(tmp_path, capsys, monkeypatch):
    """(b) both tracks licensed → both scoped canonical sentences pass."""
    import echo_bench.tools.ladder_gate as lg

    monkeypatch.setattr(
        lg,
        "evaluate_mechanism_license",
        lambda *a, **k: _v3_gate_result([_M_IMP, _M_NOISE], "both_supported"),
    )
    prereg, report, ledger = _write_gate_evidence(tmp_path)
    doc = tmp_path / "both.md"
    doc.write_text(
        _M_IMP_SCOPED + "\n\n" + _M_NOISE_SCOPED + "\n", encoding="utf-8"
    )
    rc = main([str(doc)] + _gate_args(prereg, report, ledger))
    assert rc == 0, capsys.readouterr().out


def test_v31_empty_licensed_sentences_fail_closed(tmp_path, capsys, monkeypatch):
    """(c) licensedSentences=[] → nothing licensed, even with rungs M2 True
    (boolean disguise must not reopen the v1 path)."""
    import echo_bench.tools.ladder_gate as lg

    result = _v3_gate_result([_M_IMP, _M_NOISE], "both_supported")
    result["licensedSentences"] = []  # rungs stay True — must still fail
    monkeypatch.setattr(
        lg, "evaluate_mechanism_license", lambda *a, **k: result
    )
    prereg, report, ledger = _write_gate_evidence(tmp_path)
    doc = tmp_path / "both.md"
    doc.write_text(
        _M_IMP_SCOPED + "\n\n" + _M_NOISE_SCOPED + "\n", encoding="utf-8"
    )
    rc = main([str(doc)] + _gate_args(prereg, report, ledger))
    assert rc == 1, "empty licensedSentences must license nothing (fail closed)"


def test_v31_explicit_null_licensed_sentences_fail_closed(
    tmp_path, capsys, monkeypatch
):
    """(c2) EXPLICIT ``licensedSentences: null`` (present-but-None) is MALFORMED.

    A gate result that CARRIES the ``licensedSentences`` key with a None value
    must NOT route to the v1 m2-boolean path (that would fail open: m2=True
    would license every M2-form sentence). Present-but-None is treated like any
    other malformed value: empty list, licenses nothing, Korean warning, fail
    closed. Only a genuinely ABSENT key selects the v1 behavior.
    """
    import echo_bench.tools.ladder_gate as lg

    monkeypatch.setattr(
        lg,
        "evaluate_mechanism_license",
        lambda *a, **k: {
            "rungs": {"M2": True},
            "caveatRequired": False,
            "licensedSentences": None,  # present-but-None: malformed, NOT v1
        },
    )
    prereg, report, ledger = _write_gate_evidence(tmp_path)
    doc = tmp_path / "m2.md"
    doc.write_text(_M2_CANONICAL + "\n", encoding="utf-8")
    rc = main([str(doc)] + _gate_args(prereg, report, ledger))
    out = capsys.readouterr().out
    assert rc == 1, (
        "explicit licensedSentences=None must fail closed (v3 path, empty "
        "list), not reopen the v1 m2-boolean path"
    )
    assert "[경고]" in out, (
        f"Korean malformed-licensedSentences warning expected: {out!r}"
    )


def test_v31_v1_shaped_result_preserves_old_behavior(tmp_path, capsys, monkeypatch):
    """(d) v1-shaped gate result WITHOUT licensedSentences → m2-boolean rule."""
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


def test_v31_caveat_required_interacts_with_sentence_license(
    tmp_path, capsys, monkeypatch
):
    """caveatRequired=True: the licensed sentence passes only WITH the marker."""
    import echo_bench.tools.ladder_gate as lg

    monkeypatch.setattr(
        lg,
        "evaluate_mechanism_license",
        lambda *a, **k: _v3_gate_result(
            [_M_IMP], "imprint_only_supported", caveat_required=True
        ),
    )
    prereg, report, ledger = _write_gate_evidence(tmp_path)

    ok_doc = tmp_path / "ok.md"
    ok_doc.write_text(_M_IMP_SCOPED_CAVEAT + "\n", encoding="utf-8")
    rc = main([str(ok_doc)] + _gate_args(prereg, report, ledger))
    assert rc == 0, capsys.readouterr().out

    bad_doc = tmp_path / "bad.md"
    bad_doc.write_text(_M_IMP_SCOPED + "\n", encoding="utf-8")
    rc = main([str(bad_doc)] + _gate_args(prereg, report, ledger))
    assert rc == 1, "caveatRequired without the marker must remain a finding"


def test_v31_licensed_sentence_wrapped_across_lines():
    """A canonical sentence wrapped across doc lines still matches: whitespace
    runs are normalized to single spaces on BOTH sides before the exact
    (case-sensitive) substring comparison."""
    words = _M_IMP_SCOPED.split()
    wrapped = "\n".join(
        " ".join(words[i : i + 8]) for i in range(0, len(words), 8)
    )
    findings = scan_text(
        wrapped,
        mechanism_license={
            "m2": False,
            "caveatRequired": False,
            "licensedSentences": [_M_IMP],
        },
    )
    assert findings == [], (
        f"line-wrapped licensed canonical sentence wrongly flagged: {findings}"
    )


def test_v31_licensed_sentence_match_is_case_sensitive():
    lowered = _SCOPE_PREFIX + _M_IMP.lower()
    findings = scan_text(
        lowered,
        mechanism_license={
            "m2": False,
            "caveatRequired": False,
            "licensedSentences": [_M_IMP],
        },
    )
    assert findings, "case-mangled canonical sentence must NOT be licensed"


def test_v31_paraphrase_not_licensed():
    """Near-paraphrase of the canonical sentence must NOT be licensed."""
    paraphrase = (
        "Within the tested policy families in this controlled testbed, "
        "one-step trace imprinting strongly amplifies above-null slate "
        "separability."
    )
    findings = scan_text(
        paraphrase,
        mechanism_license={
            "m2": False,
            "caveatRequired": False,
            "licensedSentences": [_M_IMP],
        },
    )
    assert findings, "paraphrased canonical sentence must NOT be licensed"


def test_v31_malformed_licensed_sentences_fail_closed():
    """licensedSentences as a bare string must license NOTHING (a string would
    otherwise iterate per-character and fail open on 1-char substrings)."""
    findings = scan_text(
        _M_IMP_SCOPED,
        mechanism_license={
            "m2": True,
            "caveatRequired": False,
            "licensedSentences": _M_IMP,  # malformed: str, not list
        },
    )
    assert findings, "malformed (non-list) licensedSentences must fail closed"


def test_v31_malformed_gate_licensed_sentences_cli_fail_closed(
    tmp_path, capsys, monkeypatch
):
    """Gate returning a malformed licensedSentences value → fail closed in CLI."""
    import echo_bench.tools.ladder_gate as lg

    monkeypatch.setattr(
        lg,
        "evaluate_mechanism_license",
        lambda *a, **k: {
            "rungs": {"M2": True},
            "caveatRequired": False,
            "licensedSentences": _M_IMP,  # malformed: str, not list
        },
    )
    prereg, report, ledger = _write_gate_evidence(tmp_path)
    doc = tmp_path / "imp.md"
    doc.write_text(_M_IMP_SCOPED + "\n", encoding="utf-8")
    rc = main([str(doc)] + _gate_args(prereg, report, ledger))
    assert rc == 1, "malformed gate licensedSentences must fail closed"


def test_v31_scope_markers_still_required_even_when_licensed():
    """A licensed sentence stripped of the scope markers is a finding.

    draftRevision 3 정렬 후 정본 문장 자체가 두 scope 마커를 포함하므로,
    마커 부재 케이스는 접두부를 구식 단수형으로 강등시켜 합성한다 — 라이선스
    문자열에 마커 없는 변형이 들어 있어도 scope 마커 요구는 우회 불가.
    """
    degraded = _M_IMP.replace(
        "Within the tested policy families in this controlled testbed, ",
        "Within the tested policy family and controlled testbed, ",
    )
    assert degraded != _M_IMP, "접두부 강등이 적용되지 않음 — 픽스처 무효"
    findings = scan_text(
        degraded,
        mechanism_license={
            "m2": True,
            "caveatRequired": False,
            "licensedSentences": [degraded],
        },
    )
    assert findings, "scope markers stay mandatory for M2-form licensing"


def test_v31_aligned_bare_canonical_passes_when_licensed():
    """draftRevision 3 정렬 후: 정본 문장 단독이 라이선스 하에 그대로 통과.

    이 정렬의 존재 이유 — 정본 문장이 자체적으로 두 scope 마커를 갖춰
    wrapper 문장 없이 출판 가능해야 한다 (무패치 통합 프로브의 수정 사항).
    """
    findings = scan_text(
        _M_IMP,
        mechanism_license={
            "m2": True,
            "caveatRequired": False,
            "licensedSentences": [_M_IMP],
        },
    )
    assert not findings, f"정렬된 정본 문장이 라이선스 하에 통과해야 함: {findings}"


def test_v31_natural_caveated_form_passes():
    """caveatRequired 하의 자연 출판형: 정본(말미 마침표 제거) + ', <caveat>.'

    caveat 마커는 같은 문장 안에 있어야 하므로 정본 문장의 마침표 앞에 붙는
    형태가 유일한 자연형이다. 매칭은 정본의 말미 마침표 1개를 벗기고 비교
    (무패치 통합 프로브가 드러낸 caveat-출판 불능 구조의 수정).
    """
    natural = _M_IMP[:-1] + ", " + _CAVEAT + "."
    findings = scan_text(
        natural,
        mechanism_license={
            "m2": True,
            "caveatRequired": True,
            "licensedSentences": [_M_IMP],
        },
    )
    assert not findings, f"자연 caveat 출판형이 통과해야 함: {findings}"
    # caveat 마커 누락 시에는 여전히 finding (caveat 의무는 불변)
    findings_no_marker = scan_text(
        _M_IMP,
        mechanism_license={
            "m2": True,
            "caveatRequired": True,
            "licensedSentences": [_M_IMP],
        },
    )
    assert findings_no_marker, "caveatRequired인데 마커 없는 정본이 통과하면 안 됨"


# --- Regression: the local TRD v3 working doc stays scanner-clean ---

_TRD_V3_PATH = _REPO_ROOT / "tasks" / "TRD_MECHANISM_V3.md"


def test_trd_mechanism_v3_scans_clean():
    """tasks/TRD_MECHANISM_V3.md must produce zero findings (when present).

    The TRD is a gitignored local working doc; skip when absent. Its N7-4
    bullet cites the measured-gap example sentence — the citation must keep a
    sentence boundary between prose and the backticked example so the
    deliberate straddle-IN anti-evasion rule does not (correctly) fire on it.
    """
    if not _TRD_V3_PATH.exists():
        pytest.skip("tasks/TRD_MECHANISM_V3.md not present (gitignored local file)")
    findings = scan_path(_TRD_V3_PATH)
    assert findings == [], (
        f"tasks/TRD_MECHANISM_V3.md must scan clean: {findings}"
    )

# src/echo_bench/tools/report_html.py
"""HTML 시각화 리포트 생성기 (ECHO-Bench).

`benchmark_index.json` 이 가리키는(= 현재 벤치마크 실행) E1/E2/E3 리포트와
`frontier_data.json` 을 읽어, 주제·설계방법·가설·실험내용·결과·결과해석·재현성을
효과적으로 시각화한 단일 자체완결 HTML 리포트를 생성한다
(`outputs/reports/echo_bench_report.html`). 차트는 Chart.js(CDN)로 렌더링한다.

가드레일: 통제된 테스트베드의 시스템 수준 결과만 제시한다. leakage 는 PROXY 로
명시하며, 사용자 선호/경험/감정/웰빙/법적(GDPR) 주장이나 실세계 일반화 주장을
하지 않는다.

식별자/키/경로는 영문, 본문/로그는 한글(프로젝트 로깅 관례)이다.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from echo_bench.logging import get_logger, log_ko

__all__ = ["generate_report_html", "main"]

_logger = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REPORTS_DIR = _REPO_ROOT / "outputs" / "reports"

_POLICY_DESC = {
    "RANDOM": "제약을 만족하는 무작위 슬레이트 (트레이스 전용)",
    "FIXED_LOW_TO_HIGH": "복잡도 낮음→높음 고정 커리큘럼 (트레이스 전용)",
    "FIXED_BALANCED": "복잡도 밴드 균형 고정 정책 (트레이스 전용)",
    "TRACE_GREEDY": "트레이스 기반 그리디(좌표 신규성) (트레이스 전용)",
    "TRACE_LIN_UCB": "트레이스 기반 선형 UCB (트레이스 전용)",
    "PSEUDO_USER_MODEL": "잠재 벡터 대조 베이스라인 (격리; 유일하게 잠재 벡터 허용)",
    "ORACLE_STRATEGY": "프로브 접근 가능한 후회 기준선 (오라클)",
}

_METRIC_DESC = [
    ("coordinate_coverage", "좌표 공간 커버리지 — 방문한 격자 셀의 비율"),
    ("artifact_diversity", "복잡도 밴드 분포의 정규화 섀넌 엔트로피"),
    ("redundancy_rate", "이전 선택/좌표를 반복한 라운드의 비율"),
    ("round_coherence", "연속 라운드 간 진행의 매끄러움(1 - 평균 변화)"),
    ("strategy_sensitivity", "통제된 프로브에 따른 관측 트레이스의 변화량"),
    ("regret_to_oracle", "C-007 오라클 기준 대비 정규화 후회(좌표 신규성)"),
    ("leakage_proxy", "관측 분포가 프로브 정체성과 공변하는 정도(PROXY; 보증 아님)"),
    ("robustness_score", "통제된 결함 하 시스템 수준 민감도"),
    ("replay_consistency", "config + seed 로부터의 정확 재현(재현 불가 시 주장 불가)"),
]


def _load(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _indexed_report(index: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    prefix = {
        "E1_HORIZON_SWEEP": "e1_horizon_",
        "E2_POLICY_UTILITY": "e2_policy_",
        "E3_AUDIT": "e3_audit_",
    }[name]
    for entry in index["experiments"]:
        if entry["experiment"] == name:
            path = _REPORTS_DIR / f"{prefix}{entry['seedBatchId'][:12]}.json"
            if path.exists():
                return _load(path)
    return None


def _build_data() -> Dict[str, Any]:
    """현재 벤치마크 실행(인덱스 기준)에서 시각화용 DATA 딕셔너리를 구성한다."""
    index = _load(_REPORTS_DIR / "benchmark_index.json")
    e1 = _indexed_report(index, "E1_HORIZON_SWEEP") or {}
    e2 = _indexed_report(index, "E2_POLICY_UTILITY") or {}
    e3 = _indexed_report(index, "E3_AUDIT") or {}
    fpath = _REPORTS_DIR / "frontier_data.json"
    frontier = _load(fpath) if fpath.exists() else {"points": [], "dataHash": ""}

    cfg = index.get("config", {})

    e1_rows = [
        {
            "H": r["H"], "policy": r["policy"],
            "coordinate_coverage": r["coordinate_coverage"],
            "artifact_diversity": r["artifact_diversity"],
            "redundancy_rate": r["redundancy_rate"],
            "round_coherence": r["round_coherence"],
        }
        for r in sorted(e1.get("table", []), key=lambda x: (x["H"], x["policy"]))
    ]

    e2_rows = []
    for r in sorted(e2.get("table", []), key=lambda x: x["policy"]):
        typ = "oracle" if r.get("isOracle") else ("대조(격리)" if r.get("isContrastBaseline") else "트레이스 전용")
        cov = r["stats"]["coordinate_coverage"]
        e2_rows.append({
            "policy": r["policy"], "type": typ,
            "coordinate_coverage": r["coordinate_coverage"],
            "artifact_diversity": r["artifact_diversity"],
            "redundancy_rate": r["redundancy_rate"],
            "round_coherence": r["round_coherence"],
            "strategy_sensitivity": r["strategy_sensitivity"],
            "regret_to_oracle": r["regret_to_oracle"],
            "cov_mean": cov["mean"], "cov_low": cov["ci_low"], "cov_high": cov["ci_high"],
            "cov_std": cov["std"], "cov_n": cov["n"],
        })

    e3_leak = [
        {"policy": r["policy"], "leakage_proxy": r["leakage_proxy"]}
        for r in sorted(e3.get("leakage", {}).get("table", []), key=lambda x: x["policy"])
    ]
    e3_rob = [
        {"fault": r["fault"], "robustness_score": r["robustness_score"],
         "faultedPoolSize": r.get("faultedPoolSize")}
        for r in sorted(e3.get("robustness", {}).get("table", []), key=lambda x: x["fault"])
    ]

    return {
        "meta": {
            "generated": "2026-06-09",
            "base_seed": cfg.get("base_seed"),
            "n": cfg.get("n"),
            "indexHash": index.get("indexHash", ""),
            "frontierDataHash": frontier.get("dataHash", ""),
        },
        "policies_desc": _POLICY_DESC,
        "metrics_desc": _METRIC_DESC,
        "e1": {"H_sweep": e1.get("config", {}).get("H_sweep", []),
               "policies": e1.get("config", {}).get("policies", []), "rows": e1_rows},
        "e2": {"H": e2.get("config", {}).get("H"), "rows": e2_rows},
        "e3": {"leakage": e3_leak, "robustness": e3_rob,
               "rob_policy": e3.get("robustness", {}).get("policy"),
               "replay": {
                   "replayable": e3.get("replayAudit", {}).get("replayable"),
                   "first_divergent": e3.get("replayAudit", {}).get("first_divergent"),
               }},
        "frontier": {"points": frontier.get("points", []), "dataHash": frontier.get("dataHash", "")},
        "index": {"experiments": index.get("experiments", []), "indexHash": index.get("indexHash", "")},
    }


def _interpretation_html(data: Dict[str, Any]) -> str:
    """데이터에 근거한 결과해석 항목(가드레일 안전)을 생성한다."""
    items: List[str] = []
    leak = data["e3"]["leakage"]
    if leak:
        hi = max(leak, key=lambda x: x["leakage_proxy"])
        lo = min(leak, key=lambda x: x["leakage_proxy"])
        items.append(
            f"<b>{hi['policy']}</b> 가 leakage_proxy 최댓값({hi['leakage_proxy']:.2f}), "
            f"<b>{lo['policy']}</b> 가 최솟값({lo['leakage_proxy']:.2f})을 보였다. "
            "이는 통제된 테스트베드에서 관측 가능한 슬레이트/선택 분포가 프로브 정체성과 "
            "공변하는 정도가 정책에 따라 달라짐을 의미한다(프록시이며 프라이버시·법적 보증이 아님)."
        )
    rows = data["e2"]["rows"]
    trace_rows = [r for r in rows if r["type"] == "트레이스 전용"]
    if trace_rows:
        hic = max(trace_rows, key=lambda x: x["cov_mean"])
        loc = min(trace_rows, key=lambda x: x["cov_mean"])
        items.append(
            f"트레이스 전용 정책 간 coordinate_coverage 평균은 <b>{loc['policy']}</b>"
            f"({loc['cov_mean']:.2f}) 부터 <b>{hic['policy']}</b>({hic['cov_mean']:.2f}) 까지 "
            "분포했다. 95% 부트스트랩 신뢰구간이 함께 보고되어 정책 간 차이의 통계적 "
            "비교 가능성을 확보한다(D-006)."
        )
    items.append(
        "PSEUDO_USER_MODEL(격리 대조 베이스라인)은 유일하게 잠재 벡터를 사용하며 "
        "트레이스 전용 정책에 영향을 주지 않는다. 따라서 트레이스 전용 정책들의 비교는 "
        "사용자 모델 없이 관측 트레이스만으로 이루어진다."
    )
    rep = data["e3"]["replay"]
    items.append(
        f"리플레이 감사 결과 replayable={rep.get('replayable')} "
        f"(first_divergent={rep.get('first_divergent')}). 동일 config+seed 재실행이 "
        "동일 reportHash 를 산출함을 확인했으며, 재현 불가능한 결과는 주된 주장을 뒷받침하지 않는다."
    )
    items.append(
        "E1 지평 스윕은 지평 H 가 커질수록 시스템 수준 지표가 어떻게 변하는지를 보여준다"
        "(아래 라인 차트). 모든 해석은 통제된 테스트베드 내부에 한정되며 실세계 일반화 주장을 하지 않는다."
    )
    return "\n".join(f"<li>{x}</li>" for x in items)


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>ECHO-Bench 시각화 리포트</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root{
    --bg:#f6f8fb; --card:#ffffff; --ink:#1f2733; --muted:#6b7785;
    --line:#e6ebf1; --accent:#4f7cff; --accent2:#23b39a; --warn:#e08a3c;
    --chip:#eef2fb;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
    font-family:-apple-system,"Apple SD Gothic Neo","Noto Sans KR",Segoe UI,Roboto,sans-serif;
    line-height:1.7}
  header.hero{background:linear-gradient(135deg,#2b3a67,#4f7cff);color:#fff;padding:48px 24px 40px}
  .wrap{max-width:1080px;margin:0 auto;padding:0 20px}
  header.hero h1{margin:0 0 6px;font-size:30px;letter-spacing:-.5px}
  header.hero p{margin:4px 0;opacity:.92}
  .meta-chips{margin-top:14px;display:flex;flex-wrap:wrap;gap:8px}
  .meta-chips span{background:rgba(255,255,255,.16);padding:5px 11px;border-radius:20px;
    font-size:12.5px;font-family:ui-monospace,Menlo,monospace}
  nav.toc{position:sticky;top:0;z-index:10;background:rgba(255,255,255,.95);
    backdrop-filter:blur(6px);border-bottom:1px solid var(--line)}
  nav.toc .wrap{display:flex;gap:6px;flex-wrap:wrap;padding:10px 20px}
  nav.toc a{font-size:13px;color:var(--muted);text-decoration:none;padding:6px 10px;border-radius:8px}
  nav.toc a:hover{background:var(--chip);color:var(--ink)}
  section{padding:34px 0;border-bottom:1px solid var(--line)}
  h2{font-size:22px;margin:0 0 4px;display:flex;align-items:center;gap:10px}
  h2 .num{background:var(--accent);color:#fff;font-size:13px;border-radius:7px;
    padding:2px 9px;font-weight:700}
  .lead{color:var(--muted);margin:2px 0 18px}
  .grid{display:grid;gap:16px}
  .g2{grid-template-columns:repeat(2,1fr)}
  @media(max-width:820px){.g2{grid-template-columns:1fr}}
  .card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;
    box-shadow:0 1px 2px rgba(20,30,60,.04)}
  .card h3{margin:0 0 10px;font-size:15px}
  .card .cap{color:var(--muted);font-size:12.5px;margin-top:8px}
  .chartbox{position:relative;height:300px}
  ul.clean{margin:6px 0 0;padding-left:20px}
  ul.clean li{margin:7px 0}
  table{width:100%;border-collapse:collapse;font-size:13px}
  th,td{padding:7px 9px;border-bottom:1px solid var(--line);text-align:center}
  th{background:#eef2f7;font-weight:700}
  td.policy,th.policy{text-align:left;font-family:ui-monospace,Menlo,monospace;font-size:12px}
  .badge{font-size:11px;border-radius:6px;padding:1px 7px}
  .b-trace{background:#e7f0ff;color:#2c5bd6}
  .b-contrast{background:#fdeede;color:#b9711f}
  .b-oracle{background:#e7f7f1;color:#1d8f78}
  .pill{display:inline-block;background:var(--chip);border-radius:7px;padding:2px 8px;
    font-family:ui-monospace,Menlo,monospace;font-size:12px;margin:2px}
  .note{background:#fff8ef;border:1px solid #f3e3cb;border-radius:10px;padding:12px 14px;
    font-size:13px;color:#80602a}
  .kvs{font-family:ui-monospace,Menlo,monospace;font-size:12px;color:var(--muted);
    word-break:break-all}
  footer{padding:26px 0 60px;color:var(--muted);font-size:12.5px}
  .guard{font-size:12.5px;color:var(--muted);margin-top:8px}
  code{background:var(--chip);padding:1px 6px;border-radius:6px;font-size:12.5px}
</style>
</head>
<body>
<header class="hero"><div class="wrap">
  <h1>ECHO-Bench — 사용자 모델 없는 적응 벤치마크</h1>
  <p>관측 가능한 상호작용 트레이스 위에서 동작하는 적응 정책의 시스템 수준 평가 · 시각화 리포트</p>
  <div class="meta-chips" id="chips"></div>
</div></header>

<nav class="toc"><div class="wrap">
  <a href="#topic">주제</a>
  <a href="#design">설계방법</a>
  <a href="#hypo">가설</a>
  <a href="#content">실험내용</a>
  <a href="#results">결과</a>
  <a href="#interp">결과해석</a>
  <a href="#repro">재현성</a>
</div></nav>

<main class="wrap">

  <section id="topic">
    <h2><span class="num">1</span> 주제</h2>
    <p class="lead">무엇을, 왜 측정하는가</p>
    <div class="card">
      <p>ECHO-Bench 는 잠재적 <b>사용자 모델</b>이 아니라 <b>관측 가능한 상호작용 트레이스</b> 위에서
      동작하는 적응 정책을 평가하는 벤치마크다. 적응의 단위는 절차적 <b>k-슬레이트 결정</b>(N개 후보
      카드 중 k개를 정해진 라운드 지평 동안 제시)이다.</p>
      <p>기여는 카드나 시각 기저 시스템 자체가 아니라, <b>사용자 모델 없는 적응을 측정·비교 가능하게
      만드는 통제된 절차적 k-슬레이트 행동 공간</b>이다. 이는 <b>utility / leakage / replayability</b>
      프론티어를 따라 평가된다.</p>
      <p class="guard">허용 주장: 시스템 수준 유틸리티, 좌표 커버리지, 산출물 다양성, 전략 민감도,
      leakage <b>프록시</b>, 강건성, 재현성. 사용자 선호·경험·감정·웰빙·법적(GDPR) 주장 및 실세계
      일반화 주장은 하지 않는다.</p>
    </div>
  </section>

  <section id="design">
    <h2><span class="num">2</span> 설계방법</h2>
    <p class="lead">통제된 행동 공간, 파라미터, 정책, 지표, 가드레일</p>
    <div class="grid g2">
      <div class="card">
        <h3>메인 파라미터</h3>
        <p><span class="pill">k=4</span><span class="pill">basis=4</span><span class="pill">pool=64</span>
        <span class="pill">H∈{4,6,8,12,16,20}</span><span class="pill" id="npill">n=?</span></p>
        <p style="font-size:13.5px">기저: B1 Branching · B2 Reaction-Diffusion · B3 Topographic fBM ·
        B4 Flow-Attractor. k=4 라운드는 최소 3종의 서로 다른 기저를 포함해야 한다.</p>
      </div>
      <div class="card">
        <h3>실험 설계 문서 관리 정책 (G-007)</h3>
        <p style="font-size:13.5px">모든 실험은 가설·고정 파라미터·시드 배치 크기 n·보고 지표·허용
        주장 범주·<code>designVersion</code> 을 명시하는 설계 항목을 가진다. 각 실행 리포트는
        designVersion 을 기록하여 configHash·reportHash 로 전파되며 결과를 설계 의도와 연결한다
        (<code>docs/</code>=설계, <code>outputs/</code>=결과).</p>
      </div>
      <div class="card">
        <h3>정책 (7종)</h3>
        <ul class="clean" id="policyList" style="font-size:13px"></ul>
      </div>
      <div class="card">
        <h3>지표 정의</h3>
        <ul class="clean" id="metricList" style="font-size:13px"></ul>
      </div>
    </div>
  </section>

  <section id="hypo">
    <h2><span class="num">3</span> 가설</h2>
    <p class="lead">각 실험이 답하는 시스템 수준 질문</p>
    <div class="grid g2">
      <div class="card"><h3>H1 · 지평 효과 (E1)</h3>
        <p>지평 H 가 길어질수록 트레이스 전용 정책의 좌표 커버리지·다양성·중복도·일관성 등
        시스템 수준 지표는 어떻게 변하는가?</p></div>
      <div class="card"><h3>H2 · 비교 가능성 (E2)</h3>
        <p>트레이스 전용 정책들이 유틸리티·전략 민감도·오라클 대비 후회에서 서로 통계적으로 구별
        가능한가? 격리된 대조 베이스라인(PSEUDO_USER_MODEL)과는 어떻게 대비되는가?</p></div>
      <div class="card"><h3>H3 · 누출 프록시 (E3)</h3>
        <p>관측 분포가 통제된 프로브 정체성과 공변하는 정도(leakage <b>proxy</b>)가 정책별로
        다른가? (프라이버시 주장이 아니라 시스템 수준 분리도 측정)</p></div>
      <div class="card"><h3>H4 · 강건성·재현성 (E3)</h3>
        <p>통제된 결함 하에서 시스템 수준 민감도는 어떠한가? 모든 실행은 config+seed 로부터
        정확히 재현되는가?</p></div>
    </div>
  </section>

  <section id="content">
    <h2><span class="num">4</span> 실험내용</h2>
    <p class="lead">실험 매트릭스와 절차</p>
    <div class="card">
      <ul class="clean" style="font-size:13.5px">
        <li><b>E1 (Horizon Sweep)</b>: 지평 H 집합을 휩쓸며 트레이스 전용 정책(RANDOM,
        FIXED_LOW_TO_HIGH, TRACE_GREEDY)의 시스템 수준 지표 변화를 측정. 셀당 n개 시드 배치 평균.</li>
        <li><b>E2 (Policy Utility)</b>: 고정 지평(H=<span id="e2H">?</span>)·풀에서 7개 정책을 4개
        트레이스 지표 + strategy_sensitivity + regret_to_oracle 로 비교. 정책당 n개 시드 배치,
        지표별 평균±95% 부트스트랩 CI.</li>
        <li><b>E3 (Audit)</b>: (a) leakage <b>proxy</b> — 프로브별 트레이스 패밀리에서 분리도 측정,
        (b) robustness — 통제된 결함(pool_shrink, basis_dropout, salience_perturb) 하 민감도,
        (c) replay 감사 — 스모크 코어 2회 재실행 후 해시 일치 검증.</li>
        <li><b>보충 S1–S4</b>: k 민감도 · 기저 절제 · 좌표 스크램블 · 살리언스 감사(별도 러너).</li>
      </ul>
      <p class="cap">프로브는 통제된 계측 입력이며 합성 사용자가 아니다. GPU 는 선택적이며 트레이스
      생성 경로에 진입하지 않는다(코어는 CPU 재현 가능).</p>
    </div>
  </section>

  <section id="results">
    <h2><span class="num">5</span> 결과</h2>
    <p class="lead">현재 실행(인덱스 고정)의 측정 결과 — 모든 차트/표는 해시된 리포트에서 파생</p>

    <div class="card" style="margin-bottom:16px">
      <h3>프론티어 — utility × leakage_proxy</h3>
      <div class="chartbox"><canvas id="frontier"></canvas></div>
      <p class="cap">x=coordinate_coverage(평균), y=leakage_proxy(<b>PROXY</b>). 각 점은 정책.
      인용 기준 산출물은 해시된 <code>frontier_data.json</code> 이다.</p>
    </div>

    <div class="grid g2">
      <div class="card">
        <h3>E2 — coordinate_coverage 평균 ± 95% CI</h3>
        <div class="chartbox"><canvas id="covci"></canvas></div>
        <p class="cap">막대=평균, 오차막대=부트스트랩 95% CI(값의 canonical_hash 로 시드 → 재현 가능).</p>
      </div>
      <div class="card">
        <h3>E2 — 정책별 지표 레이더</h3>
        <div class="chartbox"><canvas id="radar"></canvas></div>
        <p class="cap">범례를 눌러 정책을 토글. 6개 지표(모두 [0,1]).</p>
      </div>
      <div class="card">
        <h3>E1 — coordinate_coverage vs 지평 H</h3>
        <div class="chartbox"><canvas id="e1cov"></canvas></div>
        <p class="cap">정책별 라인. 셀당 n개 시드 배치 평균.</p>
      </div>
      <div class="card">
        <h3>E1 — redundancy_rate vs 지평 H</h3>
        <div class="chartbox"><canvas id="e1red"></canvas></div>
        <p class="cap">지평이 길어질 때 중복 선택 경향 변화.</p>
      </div>
      <div class="card">
        <h3>E3 — leakage_proxy (정책별)</h3>
        <div class="chartbox"><canvas id="leak"></canvas></div>
        <p class="cap"><b>PROXY</b> — 분리도 측정이며 프라이버시/법적 보증이 아님.</p>
      </div>
      <div class="card">
        <h3>E3 — robustness_score (결함별)</h3>
        <div class="chartbox"><canvas id="rob"></canvas></div>
        <p class="cap" id="robcap">통제된 결함 하 시스템 수준 민감도.</p>
      </div>
    </div>

    <div class="card" style="margin-top:16px">
      <h3>E2 — 정책 유틸리티 표 (전 지표 평균)</h3>
      <div style="overflow-x:auto"><table id="e2table"></table></div>
    </div>
  </section>

  <section id="interp">
    <h2><span class="num">6</span> 결과해석</h2>
    <p class="lead">통제된 테스트베드 내부의 시스템 수준 해석</p>
    <div class="card"><ul class="clean" id="interpList"></ul></div>
  </section>

  <section id="repro">
    <h2><span class="num">7</span> 재현성</h2>
    <p class="lead">해시 체인과 재현 보증</p>
    <div class="card">
      <p>각 실행은 configHash→archiveHash→poolHash→slateHash→traceHash→outputHash→reportHash 의
      해시 체인과 seedBatchId 를 기록한다. 동일 config+seed 재실행은 동일 reportHash 를 산출한다.
      E3 는 <code>validate_replay</code> 로 직접 검증되고, E1/E2 는 F-006 리그레션 가드로 검증된다.</p>
      <div style="overflow-x:auto"><table id="reprotable"></table></div>
      <p class="kvs" id="indexhash" style="margin-top:10px"></p>
    </div>
  </section>

</main>

<footer><div class="wrap">
  ECHO-Bench 시각화 리포트 · 통제된 테스트베드의 시스템 수준 결과 · leakage 는 PROXY ·
  실세계 일반화 주장 없음 · 그림/표는 해시된 리포트에서 결정론적으로 파생.
</div></footer>

<script>
const DATA = __DATA_JSON__;
const INTERP = `__INTERP_HTML__`;

const PALETTE = ["#4f7cff","#23b39a","#e08a3c","#d65db1","#845ec2","#2c73d2","#ff8066"];
const colorFor = (i)=>PALETTE[i % PALETTE.length];

// 칩 / 리스트
const m = DATA.meta;
document.getElementById("chips").innerHTML =
  `<span>생성일 ${m.generated}</span><span>base_seed=${m.base_seed}</span>`+
  `<span>n=${m.n}</span><span>indexHash ${String(m.indexHash).slice(0,16)}</span>`;
document.getElementById("npill").textContent = "n="+m.n;
document.getElementById("e2H").textContent = DATA.e2.H;
document.getElementById("policyList").innerHTML =
  Object.entries(DATA.policies_desc).map(([k,v])=>`<li><code>${k}</code> — ${v}</li>`).join("");
document.getElementById("metricList").innerHTML =
  DATA.metrics_desc.map(([k,v])=>`<li><code>${k}</code> — ${v}</li>`).join("");
document.getElementById("interpList").innerHTML = INTERP;
document.getElementById("robcap").textContent =
  "통제된 결함 하 시스템 수준 민감도 (감사 정책: "+DATA.e3.rob_policy+"). 실세계 분포 이동이 아님.";

// 오차막대 플러그인 (CI 차트용)
const errorBars = {
  id:"errorBars",
  afterDatasetsDraw(chart){
    const ci = chart.options.plugins.errorBars && chart.options.plugins.errorBars.ci;
    if(!ci) return;
    const {ctx, scales:{y}} = chart;
    const meta = chart.getDatasetMeta(0);
    ctx.save(); ctx.strokeStyle="#33414f"; ctx.lineWidth=1.3;
    meta.data.forEach((bar,i)=>{
      if(!ci[i]) return;
      const x=bar.x, yHi=y.getPixelForValue(ci[i].high), yLo=y.getPixelForValue(ci[i].low);
      ctx.beginPath();
      ctx.moveTo(x,yHi); ctx.lineTo(x,yLo);
      ctx.moveTo(x-6,yHi); ctx.lineTo(x+6,yHi);
      ctx.moveTo(x-6,yLo); ctx.lineTo(x+6,yLo);
      ctx.stroke();
    });
    ctx.restore();
  }
};

const unit = {min:0,max:1,ticks:{stepSize:0.2}};

// 1) 프론티어 (정책별 단일 점)
new Chart(document.getElementById("frontier"),{
  type:"scatter",
  data:{datasets: DATA.frontier.points.map((p,i)=>({
    label:p.policy, data:[{x:p.utility,y:p.leakage_proxy}],
    backgroundColor:colorFor(i), pointRadius:8, pointHoverRadius:10}))},
  options:{responsive:true,maintainAspectRatio:false,
    scales:{x:{title:{display:true,text:"utility (coordinate_coverage)"},...unit},
            y:{title:{display:true,text:"leakage_proxy (PROXY)"},...unit}},
    plugins:{legend:{position:"right",labels:{boxWidth:10,font:{size:11}}},
      tooltip:{callbacks:{label:(c)=>`${c.dataset.label}: util ${c.parsed.x.toFixed(3)}, leak ${c.parsed.y.toFixed(3)}`}}}}
});

// 2) E2 coordinate_coverage 평균 + CI
const e2 = DATA.e2.rows;
new Chart(document.getElementById("covci"),{
  type:"bar",
  data:{labels:e2.map(r=>r.policy),
    datasets:[{label:"mean coverage",data:e2.map(r=>r.cov_mean),
      backgroundColor:e2.map((_,i)=>colorFor(i)+"cc")}]},
  options:{responsive:true,maintainAspectRatio:false,
    scales:{y:{...unit,title:{display:true,text:"coordinate_coverage"}},
            x:{ticks:{font:{size:9},maxRotation:60,minRotation:35}}},
    plugins:{legend:{display:false},
      errorBars:{ci:e2.map(r=>({low:r.cov_low,high:r.cov_high}))},
      tooltip:{callbacks:{label:(c)=>{const r=e2[c.dataIndex];
        return `평균 ${r.cov_mean.toFixed(3)} [${r.cov_low.toFixed(3)}, ${r.cov_high.toFixed(3)}], n=${r.cov_n}`;}}}}},
  plugins:[errorBars]
});

// 3) E2 레이더
const radarMetrics = ["coordinate_coverage","artifact_diversity","redundancy_rate","round_coherence","strategy_sensitivity","regret_to_oracle"];
new Chart(document.getElementById("radar"),{
  type:"radar",
  data:{labels:radarMetrics.map(s=>s.replace("coordinate_","").replace("_rate","").replace("artifact_","")),
    datasets:e2.map((r,i)=>({label:r.policy,
      data:radarMetrics.map(k=>r[k]),
      borderColor:colorFor(i),backgroundColor:colorFor(i)+"22",
      borderWidth:1.5,pointRadius:2,hidden:i>=3}))},
  options:{responsive:true,maintainAspectRatio:false,
    scales:{r:{min:0,max:1,ticks:{stepSize:0.25,font:{size:9}},pointLabels:{font:{size:10}}}},
    plugins:{legend:{position:"bottom",labels:{boxWidth:9,font:{size:9.5}}}}}
});

// E1 라인 헬퍼
function e1Line(canvasId, metric){
  const Hs = DATA.e1.H_sweep;
  const pols = [...new Set(DATA.e1.rows.map(r=>r.policy))].sort();
  const ds = pols.map((p,i)=>({label:p,borderColor:colorFor(i),backgroundColor:colorFor(i),
    tension:.25,pointRadius:3,
    data:Hs.map(h=>{const row=DATA.e1.rows.find(r=>r.H===h&&r.policy===p); return row?row[metric]:null;})}));
  new Chart(document.getElementById(canvasId),{type:"line",
    data:{labels:Hs,datasets:ds},
    options:{responsive:true,maintainAspectRatio:false,
      scales:{y:{...unit,title:{display:true,text:metric}},x:{title:{display:true,text:"H (지평)"}}},
      plugins:{legend:{position:"bottom",labels:{boxWidth:10,font:{size:10}}}}}});
}
e1Line("e1cov","coordinate_coverage");
e1Line("e1red","redundancy_rate");

// 5) E3 leakage
const lk = DATA.e3.leakage;
new Chart(document.getElementById("leak"),{type:"bar",
  data:{labels:lk.map(r=>r.policy),datasets:[{label:"leakage_proxy",
    data:lk.map(r=>r.leakage_proxy),backgroundColor:lk.map((_,i)=>colorFor(i)+"cc")}]},
  options:{responsive:true,maintainAspectRatio:false,
    scales:{y:{...unit,title:{display:true,text:"leakage_proxy (PROXY)"}},
            x:{ticks:{font:{size:9},maxRotation:60,minRotation:35}}},
    plugins:{legend:{display:false}}}});

// 6) E3 robustness
const rb = DATA.e3.robustness;
new Chart(document.getElementById("rob"),{type:"bar",
  data:{labels:rb.map(r=>r.fault),datasets:[{label:"robustness_score",
    data:rb.map(r=>r.robustness_score),backgroundColor:"#23b39acc"}]},
  options:{responsive:true,maintainAspectRatio:false,
    scales:{y:{min:0,max:Math.max(0.2,...rb.map(r=>r.robustness_score))*1.2,
      title:{display:true,text:"robustness_score"}}},
    plugins:{legend:{display:false}}}});

// E2 표
const fmt=(x)=>Number(x).toFixed(3);
const badge=(t)=> t==="oracle"?'<span class="badge b-oracle">oracle</span>':
  (t==="대조(격리)"?'<span class="badge b-contrast">대조(격리)</span>':'<span class="badge b-trace">트레이스</span>');
document.getElementById("e2table").innerHTML =
  `<thead><tr><th class="policy">policy</th><th>유형</th><th>cover</th><th>divers</th>`+
  `<th>redund</th><th>coher</th><th>s.sens</th><th>regret</th></tr></thead><tbody>`+
  e2.map(r=>`<tr><td class="policy">${r.policy}</td><td>${badge(r.type)}</td>`+
    `<td>${fmt(r.coordinate_coverage)}</td><td>${fmt(r.artifact_diversity)}</td>`+
    `<td>${fmt(r.redundancy_rate)}</td><td>${fmt(r.round_coherence)}</td>`+
    `<td>${fmt(r.strategy_sensitivity)}</td><td>${fmt(r.regret_to_oracle)}</td></tr>`).join("")+
  `</tbody>`;

// 재현성 표
const rep=(v)=> v===true?"True":(v===null?"null(설계상 결정론적; F-006 검증)":String(v));
document.getElementById("reprotable").innerHTML =
  `<thead><tr><th class="policy">experiment</th><th class="policy">reportHash</th>`+
  `<th class="policy">seedBatchId</th><th>replayable</th></tr></thead><tbody>`+
  DATA.index.experiments.map(e=>`<tr><td class="policy">${e.experiment}</td>`+
    `<td class="policy">${String(e.reportHash).slice(0,16)}</td>`+
    `<td class="policy">${String(e.seedBatchId).slice(0,12)}</td>`+
    `<td>${rep(e.replayable)}</td></tr>`).join("")+`</tbody>`;
document.getElementById("indexhash").textContent =
  "indexHash: "+DATA.index.indexHash+"   ·   frontier dataHash: "+DATA.meta.frontierDataHash;
</script>
</body>
</html>
"""


def generate_report_html(out_path: Optional[Path] = None) -> str:
    """HTML 시각화 리포트를 생성하고 경로를 반환한다."""
    out_path = Path(out_path) if out_path else (_REPORTS_DIR / "echo_bench_report.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = _build_data()
    html = _TEMPLATE.replace("__DATA_JSON__", json.dumps(data, ensure_ascii=False))
    html = html.replace("__INTERP_HTML__", _interpretation_html(data))
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    log_ko(_logger, f"HTML 리포트 작성 완료: path={out_path}")
    return str(out_path)


def main() -> None:
    """CLI 진입점."""
    path = generate_report_html()
    log_ko(_logger, f"리포트 HTML: {path}")


if __name__ == "__main__":
    main()

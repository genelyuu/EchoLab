# src/echo_bench/tools/report_pdf.py
"""한글 PDF 리포트 생성기 (ECHO-Bench).

`outputs/reports/benchmark_index.json` 및 그것이 참조하는 E1/E2/E3 리포트와
프론티어 그림을 읽어, **실험 설계**와 **실험 내용(결과)** 을 모두 담은 한글
PDF 리포트를 `outputs/reports/echo_bench_report_ko.pdf` 로 생성한다.

가드레일: 본 리포트는 통제된 테스트베드에 대한 시스템 수준 진술만 포함한다.
leakage 는 PROXY 로 명시하며, 사용자 선호/경험/감정/웰빙/법적(GDPR) 주장이나
실세계 일반화 주장을 하지 않는다.

이 모듈은 `viz` 추가 의존성(matplotlib)을 필요로 한다. 식별자/키/경로는 영문,
런타임 로그·리포트 본문은 한글(프로젝트 로깅 관례)이다.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages

from echo_bench.logging import get_logger, log_ko

__all__ = ["generate_report_pdf", "main"]

_logger = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REPORTS_DIR = _REPO_ROOT / "outputs" / "reports"
_ARTIFACTS_DIR = _REPO_ROOT / "outputs" / "artifacts"

# A4 세로 (인치).
_A4 = (8.27, 11.69)

# 한글 글리프를 지원하는 후보 폰트 (macOS 기본 포함).
_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/System/Library/Fonts/Supplemental/NotoSansGothic-Regular.ttf",
    "/Library/Fonts/NanumGothic.ttf",
]

# 정책 설명 (시스템 수준, 통제된 행동 공간 관점).
_POLICY_DESC = {
    "RANDOM": "제약을 만족하는 무작위 슬레이트 (트레이스 전용)",
    "FIXED_LOW_TO_HIGH": "복잡도 낮음→높음 고정 커리큘럼 (트레이스 전용)",
    "FIXED_BALANCED": "복잡도 밴드 균형 고정 정책 (트레이스 전용)",
    "TRACE_GREEDY": "트레이스 기반 그리디(좌표 신규성) (트레이스 전용)",
    "TRACE_LIN_UCB": "트레이스 기반 선형 UCB (트레이스 전용)",
    "PSEUDO_USER_MODEL": "잠재 벡터 대조 베이스라인 (격리; 유일하게 잠재 벡터 허용)",
    "ORACLE_STRATEGY": "프로브 접근 가능한 후회 기준선 (오라클)",
}

# 지표 정의 (시스템 수준).
_METRIC_DESC = [
    ("coordinate_coverage", "좌표 공간 커버리지 — 방문한 격자 셀의 비율"),
    ("artifact_diversity", "복잡도 밴드 분포의 정규화 섀넌 엔트로피"),
    ("redundancy_rate", "이전 선택/좌표를 반복한 라운드의 비율"),
    ("round_coherence", "연속 라운드 간 진행의 매끄러움 (1 - 평균 변화)"),
    ("strategy_sensitivity", "통제된 프로브에 따른 관측 트레이스의 변화량"),
    ("regret_to_oracle", "C-007 오라클 기준 대비 정규화 후회 (좌표 신규성)"),
    ("leakage_proxy", "관측 분포가 프로브 정체성과 공변하는 정도 (PROXY; 보증 아님)"),
    ("robustness_score", "통제된 결함 하 시스템 수준 민감도"),
    ("replay_consistency", "config + seed 로부터의 정확 재현 (재현 불가 시 주장 불가)"),
]


def _select_font() -> Optional[str]:
    """한글 폰트 경로를 찾아 matplotlib 에 등록하고 폰트명을 반환한다."""
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            fm.fontManager.addfont(path)
            name = fm.FontProperties(fname=path).get_name()
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            log_ko(_logger, f"리포트 한글 폰트 등록: name={name}, path={path}")
            return name
    log_ko(_logger, "경고: 한글 폰트를 찾지 못했습니다 — 한글이 깨질 수 있습니다")
    return None


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _experiment_report(index: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    """benchmark_index 의 seedBatchId 로 해당 실험 리포트 파일을 로드한다."""
    prefix = {
        "E1_HORIZON_SWEEP": "e1_horizon_",
        "E2_POLICY_UTILITY": "e2_policy_",
        "E3_AUDIT": "e3_audit_",
    }[name]
    for entry in index["experiments"]:
        if entry["experiment"] == name:
            sb = entry["seedBatchId"][:12]
            path = _REPORTS_DIR / f"{prefix}{sb}.json"
            if path.exists():
                return _load_json(path)
    return None


def _text_page(pdf: PdfPages, title: str, blocks: List[tuple]) -> None:
    """텍스트 페이지 렌더링. blocks 는 (kind, text) 의 리스트.

    kind: 'h2'(소제목), 'body'(본문), 'bullet'(글머리), 'space'(빈 줄).
    """
    fig = plt.figure(figsize=_A4)
    fig.text(0.08, 0.95, title, fontsize=17, fontweight="bold", va="top")
    y = 0.90
    for kind, text in blocks:
        if kind == "space":
            y -= 0.018
            continue
        if kind == "h2":
            y -= 0.010
            fig.text(0.08, y, text, fontsize=12.5, fontweight="bold", va="top")
            y -= 0.030
            continue
        size = 9.5
        # Full-width Hangul glyphs are ~1 em wide, so the usable character count
        # per line is far smaller than for ASCII. Wrap conservatively.
        if kind == "bullet":
            indent, prefix, width = 0.105, "• ", 46
        else:
            indent, prefix, width = 0.08, "", 50
        wrapped = textwrap.wrap(text, width=width) or [""]
        for i, line in enumerate(wrapped):
            fig.text(indent, y, (prefix if i == 0 else "  ") + line, fontsize=size, va="top")
            y -= 0.0225
    pdf.savefig(fig)
    plt.close(fig)


def _table_page(
    pdf: PdfPages,
    title: str,
    col_labels: List[str],
    rows: List[List[str]],
    note: str = "",
    fontsize: float = 8.5,
    col_widths: Optional[List[float]] = None,
) -> None:
    """표 페이지 렌더링. col_widths 는 각 열의 폭(축 분율) 리스트(선택)."""
    fig, ax = plt.subplots(figsize=_A4)
    ax.axis("off")
    fig.text(0.08, 0.95, title, fontsize=15, fontweight="bold", va="top")
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.0, 1.5)
    for (r, c), cell in table.get_celld().items():
        if col_widths is not None and c < len(col_widths):
            cell.set_width(col_widths[c])  # set after scale so it is authoritative
        if r == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#eef2f7")
    if note:
        for i, line in enumerate(textwrap.wrap(note, width=62)):
            fig.text(0.08, 0.13 - i * 0.020, line, fontsize=8, va="top", color="#444444")
    pdf.savefig(fig)
    plt.close(fig)


def _figure_page(pdf: PdfPages, title: str, image_path: Path, caption: str) -> None:
    """기존 PNG 그림을 페이지로 삽입."""
    fig = plt.figure(figsize=_A4)
    fig.text(0.08, 0.95, title, fontsize=15, fontweight="bold", va="top")
    if image_path and image_path.exists():
        img = plt.imread(str(image_path))
        ax = fig.add_axes([0.08, 0.30, 0.84, 0.58])
        ax.imshow(img)
        ax.axis("off")
    else:
        fig.text(0.08, 0.80, "프론티어 그림을 찾을 수 없습니다.", fontsize=10, va="top")
    for i, line in enumerate(textwrap.wrap(caption, width=58)):
        fig.text(0.08, 0.26 - i * 0.022, line, fontsize=8.5, va="top", color="#444444")
    pdf.savefig(fig)
    plt.close(fig)


def _fmt(x: float, n: int = 3) -> str:
    return f"{float(x):.{n}f}"


def generate_report_pdf(out_path: Optional[Path] = None) -> str:
    """한글 PDF 리포트를 생성하고 경로를 반환한다."""
    _select_font()
    out_path = Path(out_path) if out_path else (_REPORTS_DIR / "echo_bench_report_ko.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    index = _load_json(_REPORTS_DIR / "benchmark_index.json")
    e1 = _experiment_report(index, "E1_HORIZON_SWEEP")
    e2 = _experiment_report(index, "E2_POLICY_UTILITY")
    e3 = _experiment_report(index, "E3_AUDIT")
    cfg = index.get("config", {})
    n = cfg.get("n", "?")
    base_seed = cfg.get("base_seed", "?")

    frontier_data_path = _REPORTS_DIR / "frontier_data.json"
    frontier_data = _load_json(frontier_data_path) if frontier_data_path.exists() else {}
    # Select the PNG that matches the CURRENT frontier_data by its dataHash —
    # NOT a lexicographic glob, which can grab a stale figure (filenames are
    # hash-suffixed, not time-sortable).
    frontier_png = None
    if frontier_data.get("dataHash"):
        candidate = _ARTIFACTS_DIR / f"frontier_{frontier_data['dataHash'][:12]}.png"
        if candidate.exists():
            frontier_png = candidate
    if frontier_png is None:
        pngs = sorted(_ARTIFACTS_DIR.glob("frontier_*.png"))
        frontier_png = pngs[-1] if pngs else None

    with PdfPages(str(out_path)) as pdf:
        # --- 표지 ---
        fig = plt.figure(figsize=_A4)
        fig.text(0.5, 0.70, "ECHO-Bench", fontsize=30, fontweight="bold", ha="center")
        fig.text(0.5, 0.635, "사용자 모델 없는 적응 벤치마크", fontsize=15, ha="center")
        fig.text(0.5, 0.595, "실험 설계 및 실험 결과 리포트 (한글)", fontsize=12.5, ha="center")
        fig.text(0.5, 0.50, f"생성일: 2026-06-09    base_seed={base_seed}    n={n}",
                 fontsize=10.5, ha="center", color="#333333")
        fig.text(0.5, 0.465, f"benchmark indexHash: {index.get('indexHash','')[:24]}",
                 fontsize=9, ha="center", color="#555555")
        fig.text(0.5, 0.10,
                 "통제된 테스트베드에 대한 시스템 수준 결과. leakage 는 PROXY 이며 "
                 "프라이버시/법적 보증이 아님. 실세계 일반화 주장 없음.",
                 fontsize=8.5, ha="center", color="#666666", wrap=True)
        pdf.savefig(fig)
        plt.close(fig)

        # =========================== 1부: 실험 설계 ===========================
        _text_page(pdf, "1. 실험 설계 — 개요와 기여", [
            ("h2", "1.1 개요"),
            ("body", "ECHO-Bench 는 잠재적 사용자 모델이 아니라 관측 가능한 상호작용 "
                     "트레이스 위에서 동작하는 적응 정책을 평가하는 벤치마크이다. 적응의 "
                     "단위는 절차적 k-슬레이트 결정(N개 후보 카드 중 k개를, 정해진 라운드 "
                     "지평 동안 제시)이다."),
            ("h2", "1.2 기여"),
            ("body", "기여는 카드나 시각 기저 시스템 자체가 아니라, 사용자 모델 없는 "
                     "적응을 측정·비교 가능하게 만드는 '통제된 절차적 k-슬레이트 행동 공간'이다. "
                     "이는 utility / leakage / replayability 프론티어를 따라 평가된다."),
            ("h2", "1.3 비협상 가드레일"),
            ("bullet", "트레이스 전용 정책은 user_id/인구통계/페르소나/감정/선호 벡터/자유 텍스트/"
                       "진단 라벨을 사용하지 않는다. PSEUDO_USER_MODEL 만 잠재 벡터를 갖는 격리된 대조군이다."),
            ("bullet", "사용자 선호·경험·감정·웰빙·법적(GDPR) 주장을 하지 않는다."),
            ("bullet", "코어 벤치마크는 CPU 재현 가능해야 한다. GPU 는 선택적이며 배치 카드 생성/"
                       "신경망 베이스라인에만 허용된다(트레이스 생성 경로에는 불가)."),
            ("bullet", "모든 실행은 재현성 팩을 생성한다. 재현 불가능한 결과는 주된 주장을 뒷받침할 수 없다."),
            ("bullet", "테스트베드는 의도적으로 통제된 환경이며 생태적 현실성을 주장하지 않는다."),
        ])

        _text_page(pdf, "2. 실험 설계 — 파라미터와 정책", [
            ("h2", "2.1 메인 파라미터"),
            ("bullet", "k=4, 기저(basis) 수=4, 기본 후보 풀 크기=64."),
            ("bullet", "기저: B1 Branching, B2 Reaction-Diffusion, B3 Topographic fBM, B4 Flow-Attractor."),
            ("bullet", "지평 집합 H ∈ {4, 6, 8, 12, 16, 20}; 시드 배치 크기 n="
                       f"{n} (정책당 n개 자식 시드 집계)."),
            ("bullet", "k=4 라운드는 최소 3종의 서로 다른 기저를 포함해야 한다(슬레이트 제약)."),
            ("space", ""),
            ("h2", "2.2 정책 (7종)"),
        ] + [("bullet", f"{k}: {v}") for k, v in _POLICY_DESC.items()])

        _text_page(pdf, "3. 실험 설계 — 지표 정의", [
            ("h2", "3.1 지표 (모두 [0,1] 범위, 결정론적)"),
        ] + [("bullet", f"{k}: {v}") for k, v in _METRIC_DESC] + [
            ("space", ""),
            ("body", "각 지표는 트레이스(또는 통제된 프로브/오라클 기준)의 결정론적 함수이며, "
                     "동일 입력은 동일 값을 산출한다. 시드 배치에 대해 평균/표준편차/"
                     "95% 부트스트랩 신뢰구간(CI)으로 집계된다(부트스트랩 RNG 는 값의 "
                     "canonical_hash 로 시드되어 재현 가능)."),
        ])

        _text_page(pdf, "4. 실험 설계 — 설계 관리 정책과 실험 목록", [
            ("h2", "4.1 실험 설계 문서 관리 정책 (G-007)"),
            ("body", "모든 실험은 가설, 고정 파라미터, 시드 배치 크기 n, 보고 지표, 허용 주장 "
                     "범주, designVersion 을 명시하는 설계 항목을 갖는다. 각 실행 리포트는 "
                     "designVersion 을 run_params 에 기록하여 configHash·reportHash 로 전파되며, "
                     "결과를 설계 의도와 연결한다(docs/ = 설계, outputs/ = 결과)."),
            ("space", ""),
            ("h2", "4.2 실험 목록"),
            ("bullet", "E1 (Horizon Sweep): 지평 H 를 휩쓸며 트레이스 전용 정책의 시스템 수준 "
                       "지표가 어떻게 변하는지 측정."),
            ("bullet", "E2 (Policy Utility): 고정 지평·풀에서 7개 정책을 트레이스 지표 + "
                       "strategy_sensitivity + regret_to_oracle 로 비교."),
            ("bullet", "E3 (Audit): leakage 프록시 / robustness(통제된 결함) / replay 감사."),
            ("bullet", "보충 S1–S4: k 민감도, 기저 절제, 좌표 스크램블, 살리언스 감사 (별도 러너)."),
            ("space", ""),
            ("h2", "4.3 GPU 정책 (선택적)"),
            ("body", "GPU 백엔드는 옵트인이며 화이트리스트(batch_card_generation, neural_baseline)에만 "
                     "허용된다. 해시 체인에 들어가는 GPU 산출물은 CPU 기준과 비트 일치해야 하며"
                     "(assert_cpu_equivalence, 실패 시 CPU 폴백), 트레이스 생성 경로는 GPU 를 요청하지 않는다."),
        ])

        # =========================== 2부: 실험 내용 ===========================
        # E1 표
        if e1:
            keys = ["coordinate_coverage", "artifact_diversity", "redundancy_rate", "round_coherence"]
            col = ["H", "policy", "cover", "divers", "redund", "coher", "cost"]
            rows = []
            for r in sorted(e1["table"], key=lambda x: (x["H"], x["policy"])):
                rows.append([
                    str(r["H"]), r["policy"],
                    _fmt(r["coordinate_coverage"], 2), _fmt(r["artifact_diversity"], 2),
                    _fmt(r["redundancy_rate"], 2), _fmt(r["round_coherence"], 2),
                    str(r["compute_cost"]),
                ])
            _table_page(
                pdf, "5. 실험 내용 — E1 지평 스윕 (n="
                f"{n} 평균)", col, rows,
                note="각 셀은 n개 시드 배치의 평균이다. 95% 부트스트랩 CI/표준편차는 리포트 JSON 의 "
                     "stats 블록에 있다. cost = compute_cost (= H × n). 시스템 수준, 통제된 테스트베드.",
                fontsize=8.0,
                col_widths=[0.06, 0.22, 0.115, 0.115, 0.115, 0.115, 0.08],
            )

        # E2 평균 표
        if e2:
            mkeys = ["coordinate_coverage", "artifact_diversity", "redundancy_rate",
                     "round_coherence", "strategy_sensitivity", "regret_to_oracle"]
            col = ["policy", "유형", "cover", "divers", "redund", "coher", "s.sens", "regret"]
            rows = []
            for r in sorted(e2["table"], key=lambda x: x["policy"]):
                if r.get("isOracle"):
                    typ = "oracle"
                elif r.get("isContrastBaseline"):
                    typ = "대조(격리)"
                else:
                    typ = "트레이스"
                rows.append([
                    r["policy"], typ,
                    _fmt(r["coordinate_coverage"], 2), _fmt(r["artifact_diversity"], 2),
                    _fmt(r["redundancy_rate"], 2), _fmt(r["round_coherence"], 2),
                    _fmt(r["strategy_sensitivity"], 2), _fmt(r["regret_to_oracle"], 2),
                ])
            _table_page(
                pdf, f"6. 실험 내용 — E2 정책 유틸리티 (H={e2['config']['H']}, n={n} 평균)",
                col, rows,
                note="PSEUDO_USER_MODEL 은 격리된 대조 베이스라인(유일한 잠재 벡터 정책)이며 "
                     "트레이스 전용 정책에 영향을 주지 않는다. ORACLE_STRATEGY 는 regret 기준선이다. "
                     "프로브는 통제된 계측 입력이며 합성 사용자가 아니다.",
                fontsize=7.8,
                col_widths=[0.20, 0.10, 0.095, 0.095, 0.095, 0.095, 0.095, 0.095],
            )

            # E2 CI 표 (coordinate_coverage)
            col = ["policy", "mean", "ci_low", "ci_high", "std", "n", "method"]
            rows = []
            for r in sorted(e2["table"], key=lambda x: x["policy"]):
                st = r["stats"]["coordinate_coverage"]
                rows.append([
                    r["policy"], _fmt(st["mean"]), _fmt(st["ci_low"]), _fmt(st["ci_high"]),
                    _fmt(st["std"]), str(st["n"]), st["ci_method"],
                ])
            _table_page(
                pdf, "7. 실험 내용 — E2 coordinate_coverage 신뢰구간 (D-006)",
                col, rows,
                note="시드 배치 집계: 평균 ± 95% 부트스트랩 CI. 부트스트랩은 값의 canonical_hash 로 "
                     "시드되어 재현 가능하며, 이로써 정책 간 비교의 통계적 비교 가능성을 확보한다.",
                fontsize=8.5,
                col_widths=[0.20, 0.105, 0.115, 0.115, 0.115, 0.05, 0.14],
            )

        # E3
        if e3:
            col = ["policy", "leakage_proxy", "isProxy"]
            rows = [[r["policy"], _fmt(r["leakage_proxy"]), str(r.get("isProxy", True))]
                    for r in sorted(e3["leakage"]["table"], key=lambda x: x["policy"])]
            _table_page(
                pdf, "8a. 실험 내용 — E3 누출 프록시 (leakage_proxy)",
                col, rows,
                note="leakage_proxy 는 PROXY 이다 — 관측 슬레이트/선택 분포가 통제된 프로브 정체성과 "
                     "공변하는 정도를 측정한다. 프라이버시 보증/익명성 증명/식별가능성 한계/법적 준수 주장이 아니다.",
                fontsize=9.0,
                col_widths=[0.24, 0.22, 0.14],
            )

            rcol = ["fault", "robustness_score", "faultedPoolSize"]
            rrows = [[r["fault"], _fmt(r["robustness_score"]), str(r.get("faultedPoolSize", ""))]
                     for r in sorted(e3["robustness"]["table"], key=lambda x: x["fault"])]
            replay = e3.get("replayAudit", {})
            _table_page(
                pdf, f"8b. 실험 내용 — E3 강건성/리플레이 (정책: {e3['robustness']['policy']})",
                rcol, rrows,
                note="강건성: 통제된 완전 명세 결함(pool_shrink, basis_dropout, salience_perturb) 하 "
                     "시스템 수준 민감도이며 실세계 분포 이동이 아니다.  "
                     f"리플레이 감사: replayable={replay.get('replayable')}, "
                     f"first_divergent={replay.get('first_divergent')} "
                     "(재현 불가는 숨기지 않고 보고한다).",
                fontsize=9.0,
                col_widths=[0.26, 0.22, 0.18],
            )

        # 프론티어 그림
        cap = ("utility(coordinate_coverage) 대 leakage_proxy 프론티어. 각 점은 정책이며 "
               "n=" + str(n) + " 시드 배치 평균이다. leakage 는 PROXY. ")
        if frontier_data.get("dataHash"):
            cap += f"frontier dataHash={frontier_data['dataHash'][:16]}. "
        cap += "PNG 는 편의용 렌더링이며, 인용 가능한 산출물은 해시된 frontier_data.json 이다."
        _figure_page(pdf, "9. 실험 내용 — utility×leakage 프론티어", frontier_png, cap)

        # 재현성
        repro_blocks = [
            ("h2", "10.1 재현성 해시 체인"),
            ("bullet", f"benchmark indexHash: {index.get('indexHash','')}"),
        ]
        for e in index["experiments"]:
            rep = "True" if e.get("replayable") is True else (
                "null(설계상 결정론적; F-006 가드로 검증)" if e.get("replayable") is None else str(e.get("replayable")))
            repro_blocks.append(("bullet", f"{e['experiment']}: reportHash={e['reportHash'][:16]}, "
                                            f"seedBatchId={e['seedBatchId'][:12]}, replayable={rep}"))
        repro_blocks += [
            ("space", ""),
            ("h2", "10.2 재현 보증"),
            ("body", "각 실행은 configHash→archiveHash→poolHash→slateHash→traceHash→outputHash→"
                     "reportHash 의 해시 체인과 seedBatchId 를 기록한다. 동일 config+seed 재실행은 "
                     "동일 reportHash 를 산출한다(검증됨: indexHash 재실행 일치). E3 는 "
                     "validate_replay 로 직접 검증되며, E1/E2 는 F-006 리그레션 가드로 검증된다."),
            ("space", ""),
            ("body", "참고: 모든 그림/표는 해시된 리포트에서 결정론적으로 파생된다. PNG 바이트는 "
                     "matplotlib 버전에 따라 달라질 수 있어 주된 주장의 근거가 아니며, "
                     "해시된 데이터 산출물(frontier_data.json, *_report JSON)이 인용 기준이다."),
        ]
        _text_page(pdf, "10. 재현성 (Reproducibility)", repro_blocks)

    log_ko(_logger, f"한글 PDF 리포트 작성 완료: path={out_path}")
    return str(out_path)


def main() -> None:
    """CLI 진입점."""
    path = generate_report_pdf()
    log_ko(_logger, f"리포트 PDF: {path}")


if __name__ == "__main__":
    main()

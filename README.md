# ECHO-Bench

ECHO-Bench는 **사용자 모델에 의존하지 않는 적응(User-Model-Free Adaptation) 벤치마크**입니다.
잠재적 사용자 모델을 가정하지 않고, **관측 가능한 상호작용 트레이스(observable interaction trace)** 위에서
동작하는 적응 정책을 평가합니다. 적응의 단위는 절차적으로 생성된 **k-슬레이트(k-slate)** 결정입니다.
매 라운드 N개의 후보 카드 중 k개를 제시하며, 이를 정해진 라운드 수(horizon)만큼 반복합니다.

이 저장소는 **순수 실험 코드**만 포함합니다(패키지 소스, 테스트, 설정). 생성된 산출물은 설정과 시드로부터
재현 가능하므로 버전 관리하지 않습니다.

## 핵심 기여

이 벤치마크의 기여는 카드나 시각적 베이시스 시스템 자체가 아닙니다. 그것들은 통제된 **행동 공간(action space)** 입니다.
기여는 **사용자 모델 없는 적응을 측정 가능하고 비교 가능하게 만드는, 통제된 절차적 k-슬레이트 행동 공간** 이며,
이를 **유틸리티 / 누출(leakage) / 재현성(replayability)** 의 프론티어 위에서 평가한다는 점입니다.

테스트베드는 의도적으로 통제된 환경이며, 생태학적으로 현실적인 환경이 아닙니다. 행동 공간을 통제함으로써
의미적 교란을 줄입니다(semantic confound reduction).

## 주요 개념

- **베이시스(B1~B4)**: B1 분기 구조(Branching), B2 반응-확산(Reaction-Diffusion),
  B3 지형 fBM(Topographic fBM), B4 흐름-끌개(Flow-Attractor). 카드는 `basis / seed / params` 만으로
  생성되며, 개인·사용자·의미 관련 필드는 스키마 어디에도 포함하지 않습니다.
- **트레이스(Trace)**: 라운드마다의 슬레이트와 선택을 기록한 관측 가능한 시퀀스. 잠재 변수 필드가 없습니다.
- **전략 프로브(Strategy probe)**: 계측된 입력 정책이며, 합성 사용자가 아닙니다.

## 정책

`RANDOM`, `FIXED_LOW_TO_HIGH`, `FIXED_BALANCED`, `TRACE_GREEDY`, `TRACE_LIN_UCB`,
`PSEUDO_USER_MODEL`, `ORACLE_STRATEGY`.

트레이스 전용 정책은 `user_id`, 인구통계, 페르소나, 선호 벡터, 자유 텍스트 등 어떤 잠재 정보도 사용하지 않습니다.
`PSEUDO_USER_MODEL` 만이 잠재 벡터를 가지며, 오직 대비용 기준선(contrast baseline)으로 격리되어 존재합니다.

## 주요 파라미터

- `k=4`, 베이시스 수 `=4`, 기본 후보 풀 크기 `=64`.
- `k=4` 라운드는 최소 3개의 서로 다른 베이시스를 포함합니다(`k=6`은 ≥3, 가급적 4; `k=2`는 서로 다른 두 베이시스).

## 실험

- **E1 — Horizon Sweep**: 라운드 수 변화에 따른 트레이스 전용 정책의 거동.
- **E2 — Policy Utility**: 전체 정책의 유틸리티 비교(유틸리티/누출/재현성 프론티어).
- **E3 — Leakage / Robustness / Replay 감사**: 누출 프록시, 통제된 결함 변환에 대한 강건성, 재현 검증.
- **보조 실험 S1~S4**: k 민감도, 베이시스 절제(ablation), 좌표 스크램블, 살리언스 감사.

## 저장소 구조

```
src/echo_bench/   소스. 하위 패키지: basis, cards, archive, env, policies, probes, metrics, experiments, logging, tools, utils.
configs/          basis/, archive/, policies/, experiments/, hardware/ — 설정과 시드로부터 재현 가능.
tests/            pytest 기반 테스트.
outputs/          runs/, reports/, artifacts/ — 실행 시 생성되는 산출물(버전 관리 제외, 모든 산출물은 해시로 기록).
```

## 설치

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

요구 사항: Python 3.10 이상. 핵심 의존성은 `numpy`, `pyyaml` 입니다.

## 실행

```bash
pytest                                      # 전체 테스트
python -m echo_bench.experiments.smoke      # 최소 엔드투엔드(스모크) 실행
python -m echo_bench.experiments.run_all    # 전체 벤치마크 실행
```

각 실험은 `python -m echo_bench.experiments.<이름>` 형태로 실행합니다
(`e1_horizon`, `e2_policy`, `e3_audit`, `s1_k_sensitivity`, `s2_basis_ablation`,
`s3_coordinate_scramble`, `s4_salience_audit`). 모든 실행은 `--dry-run` 모드를 지원하며,
시드 배치 id, 메트릭 테이블, 리포트 json 을 생성합니다.

## 재현성

해시는 일급 객체입니다. 각 실행은 설정 해시, 코드 커밋 해시, 후보 아카이브/풀 해시, 선택 슬레이트 해시,
트레이스 해시, 출력 해시, 리포트 해시를 기록합니다. 재현 검증기는 설정과 시드만으로 실행을 재구성합니다.
재현할 수 없는 결과는 핵심 주장을 뒷받침할 수 없습니다. 핵심 벤치마크는 CPU만으로 재현 가능하며,
GPU는 대규모 배치 생성이나 신경망 기준선에 한해 선택적으로만 사용합니다.

## 허용되는 주장

시스템 수준의 유틸리티, 좌표 커버리지, 산출물 다양성, 전략 민감도, 누출 프록시, 강건성, 재현성에 한합니다.
사용자 선호·경험·감정·웰빙이나 법적/규정 준수에 대한 주장은 하지 않으며, 현실 세계로의 일반화도 주장하지 않습니다.

## 라이선스

추후 명시 예정.

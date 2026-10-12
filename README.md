# 화장실 통합 판정 모델 (restroom_sensor_ai)

화장실에 다는 센서 상자 하나(DHT11 온습도 + MQ-135 + MQ-7)의 값으로 **16개 항목을 한 모델로 판정**합니다. 흡연, 냄새, 청소, 손 씻기, 수도, 누수, 젖은 바닥, 문 열림, 창문 열림, 환기, 동파 위험을 15초마다 판정하고, 관리자가 볼 등급(긴급 / 주의 / 정보 / 평상시)으로 정리합니다.

이 저장소에는 다음이 모두 들어 있습니다.

| 무엇 | 어디 |
|---|---|
| 학습된 모델 (16개 분류기, 임계값, 흡연 알림 규칙) | [`models/unified/`](models/unified/) |
| 알고리즘 설명, 튜닝한 값과 방법, 앞으로 튜닝하는 방법 | [`docs/ALGORITHMS.md`](docs/ALGORITHMS.md) |
| 학습 과정 (환경, 순서, 실제 실행 기록, 다시 만드는 법) | [`docs/TRAINING.md`](docs/TRAINING.md) |
| 학습 데이터 (장치 240대 × 14일, 15초 행 1,935만 개, 그래프 7장) | [`data/unified_world_v1/`](data/unified_world_v1/README.md) |
| 코드 (데이터 생성, 특징, 학습, 판정, 검사, 그래프) | [`restroom_unified/`](restroom_unified/) |

> **모든 데이터와 성능은 합성(시뮬레이션) 결과입니다. 실제 화장실 측정은 0행입니다.** 실제 설치 환경에서는 성능이 다를 수 있습니다. 특히 문 열림이 온습도에 주는 영향, 냄새·청소제가 MQ-7을 얼마나 올리는지는 공학적 가정입니다.

## 성능 (처음 보는 시험 장치 40대, 합성)

| 수준 | 항목 (F1, 15초 행 단위) |
|---|---|
| 잘 됨 | 누수 0.86, 동파 1시간 0.85, 이용·냄새 0.82, 손 씻기·습기 0.74, 흡연 후 잔류 0.74 |
| 보통 | 청소 제품 0.69, 동파 3시간 0.69, 청소+젖은 바닥 0.58, 소독제 0.56, 젖은 바닥 0.54, 수도 사용 0.54 |
| 약함 | 문 열림 0.40, 창문 열림 0.36, 환기 변화 0.31, 방향제 0.09 |

| 흡연 (사건 단위) | 감지율 | 오경보 |
|---|---|---|
| 흡연 의심 이상 (주의 + 긴급) | 76.2% | 장치당 약 6일에 1번 |
| 흡연 경보 (긴급) | 57.4% | 장치당 약 20일에 1번 |

학습과 조건이 다른 장치(추운 곳, 센서 감도 차이, 연기 약함 등)에서의 결과와 온습도만 쓸 때의 비교는 [ALGORITHMS.md 6장](docs/ALGORITHMS.md#6-결과)에 있습니다.

## 어떻게 만들었나 (요약)

1. **통합 합성 데이터:** 화장실 하나마다 온도·습도·가스 두 종류를 물리식(1차 혼합 모형)으로 15초마다 계산하고, 사건(이용, 손 씻기, 흡연, 청소, 문·창문, 누수, 난방 고장 등)과 센서(DHT11 정수·범위, MQ 센서 전도도, MQ-7 히터 주기)를 모형화했습니다. 16개 항목의 정답이 같은 시계열에 함께 있습니다.
2. **특징 56개:** 과거 값만 쓰는 이동 창 통계. MQ 센서는 전도도로 바꾸고 깨끗한 공기 기준선 대비 상승률을 봅니다. 흡연은 "냄새 대비 CO 비율"과 기기 자기 보정으로 구분합니다.
3. **모델:** scikit-learn `HistGradientBoostingClassifier`를 항목마다 하나씩(16개) 학습했습니다. 학습률 0.08, 잎 노드 63(15·31·63 비교), 잎 최소 100행, L2 1.0, 클래스 가중치 balanced.
4. **튜닝:** 반복 횟수는 항목마다 보정 장치의 평균 정밀도가 가장 좋은 지점(80~400번)에서 잘랐습니다. 임계값은 F1 최대(경보)와 정밀도 50% 이상 중 가장 민감한 값(주의)으로 골랐습니다. 흡연 알림은 "최근 n번 중 m번" 지속 규칙을 오경보 상한 아래에서 골랐습니다.
5. **평가:** 학습·보정에 쓰지 않은 시험 장치 40대와 조건을 바꾼 장치 40대에서 한 번만 평가했습니다.

자세한 내용은 [docs/ALGORITHMS.md](docs/ALGORITHMS.md)와 [docs/TRAINING.md](docs/TRAINING.md)를 보세요.

## 쓰는 법

```sh
pip install -r requirements.txt
python -m restroom_unified.check                                # 자체 검사 12개 (약 3분)
python -m restroom_unified.demo --start-hour 8 --hours 2        # 합성 장치 하나를 재생해 판정과 정답을 나란히 출력
```

```python
from datetime import datetime
from restroom_unified.model import UnifiedJudge

judge = UnifiedJudge("restroom-101")                  # models/unified/unified_model.joblib을 불러옴
# 15초마다 한 번. mq7_adc는 MQ-7 히터 주기 중 읽을 수 있는 행에서만 값, 아니면 None
result = judge.add(datetime.now(), temperature=21, humidity=48, mq135_adc=1850, mq7_adc=None)
print(result["level"], result["headline"], result["labels"])   # 처음 2시간은 "준비 중"
```

ESP32처럼 5초마다 값이 오면 `judge.add_5s(...)`를 쓰면 됩니다. 3행마다 판정 하나를 돌려줍니다.

결과에는 등급, 대표 문구, 켜진 항목, 항목별 확률, 흡연 경보·의심, 동파 단계(경보/주의)가 들어 있습니다.

## 다시 학습하기

```sh
python -m restroom_unified.train            # 데이터 생성부터 시험까지 (약 50분, 4코어)
python -m restroom_unified.retune           # 임계값·흡연 규칙만 다시 고르기 (약 20분)
python -m restroom_unified.export_dataset --out data/unified_world_v1
python -m restroom_unified.plot_dataset --data data/unified_world_v1 --out data/unified_world_v1/graphs
```

## 폴더 구조

```
restroom_unified/
  world.py            통합 합성 세계 (물리 상태, 사건, 센서, 정답 16개)
  features.py         특징 56개 (배치·실시간 동일)
  pipeline.py         데이터 묶기, 채점, 임계값·흡연 규칙 선택, 평가 지표
  train.py            학습 전체 (잎 노드 격자, 반복 횟수 선택, 임계값, 시험)
  retune.py           재학습 없이 임계값·흡연 규칙 다시 고르기
  model.py            UnifiedModel (모델 묶음), UnifiedJudge (장치 하나, 실시간 판정)
  rules.py            등급 규칙 (긴급/주의/정보/평상시)
  check.py            자체 검사
  demo.py             합성 장치 재생
  export_dataset.py   학습 데이터 내보내기
  load_dataset.py     학습 데이터 읽기
  plot_dataset.py     학습 데이터 그래프
models/unified/
  unified_model.joblib       모델 (15MB)
  policy.json                임계값·흡연 규칙
  UNIFIED_EVALUATION.json    튜닝 값과 시험 결과 전체
  CHECK_RESULT.json          자체 검사 결과
  logs/                      실제 학습·재선택 로그
data/unified_world_v1/       학습 데이터 (npz 4개, 장치 파라미터, 표본 CSV, 그래프 7장)
docs/                        ALGORITHMS.md, TRAINING.md
```

## 한계

- 합성 데이터로만 학습·평가했습니다. 실측 데이터로 임계값부터 다시 고르는 것을 권합니다([ALGORITHMS.md 7.8절](docs/ALGORITHMS.md)).
- DHT11은 0℃ 아래를 읽지 못합니다. 전자담배·궐련형 전자담배는 CO가 거의 없어 잡지 못합니다.
- MQ-7 값이 없으면 흡연을 거의 판정하지 못합니다(히터 전환 회로 필요).
- 문 열림·창문 열림·환기 변화·방향제는 아직 약합니다. 문 접점 센서 같은 직접 측정을 더하면 크게 좋아질 수 있습니다.

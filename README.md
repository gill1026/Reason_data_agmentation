# Reason Data Augmentation

DriveLM의 **6방향 이미지 + 질문(Q) + 기존 정답(A)**을 GPT-4o에 함께 전달하여, 정답을 설명하는 `thinking` 필드를 추가하는 데이터 증강 도구입니다. 같은 입력에 `detail="low"`와 `detail="high"`를 적용하여 생성 결과를 비교할 수 있습니다.

ReasonDrive의 아이디어를 참고한 별도 구현이며, ReasonDrive 저자의 공식 생성 코드나 동일한 실험을 재현한 코드는 아닙니다.

## 증강 방식

```text
DriveLM QA JSON + 동일 시점의 6방향 이미지
    → 전체 시야를 유지하며 512×288로 축소
    → 이미지 + 질문 + 기존 정답 + 범주별 지침 구성
    → 동일 입력으로 GPT-4o low / high 각각 호출
    → 원본 필드를 유지하고 thinking만 추가한 새 JSON 저장
```

- 기존 정답을 신뢰하는 데이터 주석으로 사용합니다. 정답을 다시 판정하거나 변경하지 않습니다.
- 질문이 지정한 객체와 카메라를 찾고, 관련된 시각적 특징·위치·객체 간 관계를 정답과 연결합니다.
- 인지·예측·계획·자차 행동 범주에 맞는 간결한 영어 설명을 2~3문장 생성합니다.
- 6장은 연속 영상이 아닌 **동일 시점의 서로 다른 방향 이미지**입니다.
- 이동·속도처럼 단일 시점으로 확인하기 어려운 상태는 정답에서 제공된 정보로 사용합니다. 필요하면 “According to the provided annotation”으로 출처를 구분하며, 관찰하지 않은 움직임이나 원인을 만들어내지 않도록 지시합니다.
- `thinking`은 생성된 설명이며, 모델의 숨겨진 내부 사고 과정을 추출한 것이 아닙니다.
- 이 증강은 기존 QA에 설명을 추가하는 작업입니다. 새로운 장면·질문·정답이 생성되는 것은 아닙니다.

## 파일 구성

```text
.
├── generate_think.py       # 입력·축소·프롬프트·호출·저장
├── selected_30.json        # 사용자가 별도로 준비하는 QA JSON (Git 제외)
├── requirements.txt
├── README.md
└── select_30_image/        # 사용자가 별도로 준비하는 이미지 (Git 제외)
    ├── CAM_FRONT/
    ├── CAM_FRONT_LEFT/
    ├── CAM_FRONT_RIGHT/
    ├── CAM_BACK/
    ├── CAM_BACK_LEFT/
    └── CAM_BACK_RIGHT/
```

QA JSON과 이미지는 저장소에 포함하지 않습니다. 기본 실행을 위해 QA JSON을 `selected_30.json`이라는 이름으로 준비하세요. 각 QA의 `image_paths`에 있는 **파일명**을 해당 카메라 하위 폴더에 배치하세요. JSON에 기록된 상위 경로 대신 `--images` 아래의 `카메라명/파일명`으로 실제 파일을 찾습니다. 원본 데이터의 이용·배포 조건을 확인하고 준비하세요.

## 실행 환경 및 설치

- Python 3.9 이상
- macOS: 현재 이미지 처리는 macOS 내장 `sips` 명령을 사용합니다. Windows/Linux는 그대로 실행할 수 없습니다.
- OpenAI API 키와 사용 가능한 API 잔액

```bash
git clone https://github.com/gill1026/Reason_data_agmentation.git
cd Reason_data_agmentation
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

macOS 기본 zsh에서 키를 화면에 표시하지 않고 입력합니다. 첫 줄 실행 후 나타나는 안내에 **키만 붙여넣고 Enter**를 누르세요.

```zsh
read -rs 'OPENAI_API_KEY?OpenAI API 키 입력: '
export OPENAI_API_KEY
printf '\n'
```

키를 코드나 저장소에 넣지 마세요.

## 입력 형식

입력은 JSON 배열이며, 각 항목에는 고유한 `sample_id`, 지원 범주, 질문, 정답, 6개 카메라 경로가 필요합니다. 추가 필드는 결과에 그대로 보존됩니다.

```json
[
  {
    "sample_id": "example_001",
    "scene_id": "scene-token",
    "frame_id": "frame-token",
    "category": "perception",
    "question": "What is the status of the specified vehicle?",
    "answer": "Moving.",
    "image_paths": {
      "CAM_FRONT": "samples/CAM_FRONT/front.jpg",
      "CAM_FRONT_LEFT": "samples/CAM_FRONT_LEFT/front_left.jpg",
      "CAM_FRONT_RIGHT": "samples/CAM_FRONT_RIGHT/front_right.jpg",
      "CAM_BACK": "samples/CAM_BACK/back.jpg",
      "CAM_BACK_LEFT": "samples/CAM_BACK_LEFT/back_left.jpg",
      "CAM_BACK_RIGHT": "samples/CAM_BACK_RIGHT/back_right.jpg"
    }
  }
]
```

지원 범주는 `perception`, `prediction`, `planning`, `behavior`입니다. 현재 축소 코드는 16:9 이미지만 허용합니다. 객체 ID의 좌표는 원본 기준으로 유지하며, 프롬프트에 축소 비율을 안내합니다. 예를 들어 1600×900 → 512×288이면 x와 y에 각각 0.32를 곱해 위치를 찾습니다.

## 실행

먼저 이미지 경로와 축소 처리를 확인합니다. API 호출이나 결과 JSON 생성은 하지 않지만, 축소 이미지 캐시는 생성될 수 있습니다.

```bash
python3 generate_think.py --limit 30 --dry-run
```

동일 QA 30개에 low/high를 각각 적용합니다. 최대 60회 호출하며 순차적으로 실행합니다.

```bash
python3 generate_think.py --limit 30
```

한 설정만 실행하거나 다른 입력을 사용할 수도 있습니다.

```bash
python3 generate_think.py --limit 5 --detail low
python3 generate_think.py --limit 5 --detail high
python3 generate_think.py --input my_qa.json --images /path/to/images --limit 30
```

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--limit` | `30` | 입력 앞부분에서 처리할 QA 개수. 입력 개수를 초과할 수 없음 |
| `--detail` | `both` | `both`, `low`, `high` |
| `--model` | `gpt-4o-2024-08-06` | 호출할 모델 ID |
| `--input` | 코드 옆 `selected_30.json` | QA JSON |
| `--images` | 코드 옆 `select_30_image` | 카메라별 이미지 폴더 |
| `--dry-run` | 비활성 | 입력·축소만 확인 |

low/high는 서로 다른 모델이 아니라 이미지 입력의 처리 설정입니다. 두 설정에는 동일한 512×288 이미지와 질문·정답·프롬프트를 사용합니다. 생성 설정은 `temperature=0`, `max_tokens=700`이며, 결과가 완전히 결정적임을 보장하지는 않습니다.

## 프롬프트와 출력

`generate_think.py`의 `SYSTEM`에 공통 지침, `GUIDANCE`에 범주별 지침, `build_prompt()`에 질문·정답·좌표 안내가 있습니다. 정확한 영어 프롬프트는 코드와 실행별 `manifest.json`에서 확인할 수 있습니다.

API에는 아래 항목만 있는 JSON을 요청합니다.

```json
{
  "thinking": "<think>A short explanation grounded in the supplied images, question, and annotated answer.</think>"
}
```

최종 `think_low.json`, `think_high.json`에는 **원본 항목의 모든 필드 + `thinking`**을 저장합니다. 원본 QA JSON과 원본 이미지는 수정하지 않습니다.

## 결과 및 재실행

```text
resized_512x288/                   # 이미지 내용 해시별 축소본
└── ...
gpt4o_comparison/<실험ID>/
├── manifest.json                 # 입력·프롬프트·설정·이미지 해시
├── responses.jsonl               # 성공 응답·사용량·추정 비용
├── think_low.json                # low 결과
├── think_high.json               # high 결과
├── summary.json                  # 완료 개수·추정 비용
├── invalid_responses.jsonl       # 응답 형식 오류 등이 있을 때
└── errors.jsonl                  # API 호출 오류가 있을 때
```

- `both`에서는 두 설정 모두 성공한 QA만 최종 파일에 내보내므로 low/high의 대상과 개수가 같습니다. 한쪽만 성공한 응답도 기록에 보존됩니다.
- 동일 입력·프롬프트·설정의 같은 실험을 다시 실행하면 성공 응답을 재사용하고 미완료 항목만 호출합니다.
- 프롬프트·입력·모델·선택 QA 수 등이 바뀌면 새 실험 ID가 만들어질 수 있습니다. 기존 결과는 유지되지만 새 실험에서 이전 응답을 자동으로 가져오지는 않습니다.
- 응답은 저장 전에 JSON 구조와 태그를 확인합니다. 이 검사는 내용의 사실성을 보장하지 않습니다.
- 현재 버전은 정답 타당성 자체 평가를 생성하지 않습니다. 이전 버전의 `review_*.json`, `supported_candidates_*.json`은 이전 실험의 자료입니다.

## 비용과 검수

실제 실행은 유료 API 호출입니다. 코드는 사용 토큰에 입력 $2.50/백만 토큰, 출력 $10.00/백만 토큰을 적용하여 비용을 추정합니다. 이 값은 코드에 고정된 계산 기준으로, 최신 요금·캐시 할인·실패 요청·세금을 모두 반영한 청구액은 아닙니다. 최종 사용액은 제공업체의 결제 내역에서 확인하세요.

생성 결과를 학습에 사용하기 전에는 질문 대상 식별, 이미지 근거, 객체 간 관계, 정답과 설명의 일치, 관찰하지 않은 움직임·의도·인과관계의 추가 여부를 확인해야 합니다. 기존 정답을 신뢰하도록 지시하므로, 이 코드는 원본 정답의 오류를 검출하는 도구가 아닙니다.

ReasonDrive와의 비중복은 별도의 데이터 선정 단계에서 scene/frame ID로 확인해야 합니다. 현재 생성 코드에는 ReasonDrive 원본과 대조하여 겹치는 샘플을 제외하는 기능이 없습니다.

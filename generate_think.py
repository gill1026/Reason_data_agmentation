"""같은 DriveLM QA를 512×288 이미지로 GPT-4o low/high 비교 생성.

설치: python3 -m pip install openai
준비: OPENAI_API_KEY 환경변수 설정
입력 점검: python3 generate_think.py --limit 30 --dry-run
실제 생성: python3 generate_think.py --limit 30
원본 JSON·이미지는 수정하지 않습니다. macOS sips로 별도 축소본을 만듭니다.
ReasonDrive 방식 참고 구현이며 저자의 원본 코드가 아닙니다.
"""
import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

CAMERAS = ('CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')
GUIDANCE = {
    'perception': 'Connect the queried object, its visible position and features to the annotated state in the answer.',
    'prediction': 'Connect relevant object positions and interactions to the prediction given in the answer. Do not describe future events as already observed.',
    'planning': 'Connect visible hazards and traffic constraints to the action given in the answer.',
    'behavior': 'Connect relevant road conditions and surrounding objects to the ego state or behavior given in the answer.',
}
# 공통 프롬프트: 정답을 신뢰하는 주석으로 사용하고 이미지·질문과 연결된 설명을 생성합니다.
SYSTEM = (
    'You write concise explanatory rationales for autonomous-driving training data. '
    'Use all three inputs together: six camera images, the question, and the provided reference answer. '
    'Treat the reference answer as a trusted dataset annotation and use it as the premise for the explanation. '
    'Do not re-evaluate its correctness, replace it with another answer, or conclude that the answer is unknown. '
    'First identify the object and camera referred to in the question. '
    'Connect relevant visible object features, positions and relationships to the meaning of the provided answer. '
    'Preserve the exact subject and object-to-object relationship asked about; do not substitute another object or the ego vehicle. '
    'Avoid unrelated scene descriptions and do more than merely repeat the answer when relevant visual context is available. '
    'Do not invent objects, actions, intentions, speeds, distances, calculations, or causal evidence. '
    'The six images are simultaneous views in different directions, NOT a temporal video sequence. '
    'Do not claim to observe motion across these views or to simulate maneuvers. '
    'When the answer supplies motion, speed, or behavior that cannot be directly established from a single timestamp, '
    'use that annotated state as a premise and, when needed, say "According to the provided annotation" to distinguish it from visual observation. '
    'Do not manufacture a visual proof or an unprovided cause for that state. '
    'If a target cannot be visually resolved, avoid inventing its appearance or position; '
    'keep the explanation limited to supplied annotations and relevant context that is actually visible. '
    'Write 2-3 concise English sentences with a conclusion consistent with the provided answer. '
    'Preserve object IDs exactly when used. '
    'Return only the requested JSON with a thinking field containing one nonempty <think>...</think> block. '
    'This is a brief public explanation, not hidden internal reasoning. '
    'Do not output correctness classifications or separate review comments.'
)
SCHEMA = {
    'type': 'object',
    'properties': {'thinking': {'type': 'string'}},
    'required': ['thinking'],
    'additionalProperties': False,
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(data):
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def append_record(path, record):
    # API 응답을 먼저 기록합니다. 이후 결과 저장 중 중단돼도 응답을 재사용할 수 있습니다.
    with path.open('a', encoding='utf-8') as file:
        file.write(json.dumps(record, ensure_ascii=False) + '\n')
        file.flush()
        os.fsync(file.fileno())


def dimensions(path):
    result = subprocess.run(['sips', '-g', 'pixelWidth', '-g', 'pixelHeight', str(path)],
                            check=True, capture_output=True, text=True)
    return tuple(int(re.search(label + r':\s*(\d+)', result.stdout).group(1))
                 for label in ('pixelWidth', 'pixelHeight'))


def resize_image(source, cache):
    # 1. 원본 이미지 읽기 및 축소: 원본은 보존하고, 내용 해시별로 별도 JPG를 만듭니다.
    raw_hash = digest(source.read_bytes())
    target = cache / (raw_hash + '_512x288.jpg')
    width, height = dimensions(source)
    if width * 9 != height * 16:
        raise ValueError(f'16:9 이미지가 아닙니다. 왜곡 방지를 위해 중단: {source.name}')
    if not target.exists() or dimensions(target) != (512, 288):
        cache.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.stem + '.tmp.jpg')
        subprocess.run(['sips', '-s', 'format', 'jpeg', '-z', '288', '512', str(source),
                        '--out', str(temporary)], check=True, capture_output=True)
        if dimensions(temporary) != (512, 288):
            raise ValueError('이미지 축소 크기 확인 실패')
        temporary.replace(target)
    resized = target.read_bytes()
    return dict(camera=None, original_size=[width, height], original_sha256=raw_hash,
                resized_sha256=digest(resized), resized_path=str(target)), resized


def build_prompt(sample, image_info):
    # 2. 프롬프트: 범주별 지침과 원래 좌표/축소 이미지의 좌표 관계도 전달합니다.
    sizes = {x['camera']: x['original_size'] for x in image_info}
    return (f"Category: {sample['category']}\nQuestion: {sample['question']}\n"
            f"Reference answer: {sample['answer']}\nTask guidance: {GUIDANCE[sample['category']]}\n"
            'Images are resized to 512x288 while preserving the entire 16:9 field of view. '
            'Object IDs such as <c1,CAM_FRONT,x,y> retain ORIGINAL pixel coordinates. '
            'To locate them in the resized image multiply x by 512/original_width and '
            'y by 288/original_height. Do not change those IDs in text. '
            f'Original sizes per camera: {json.dumps(sizes)}')


def normalize_payload(payload):
    # 태그는 저장 형식이므로 정상 설명 본문에는 코드에서 붙입니다.
    # 본문의 객체 ID는 보존하고, 깨진 think 태그는 자동 수정하지 않습니다.
    if isinstance(payload, dict) and isinstance(payload.get('thinking'), str):
        payload = dict(payload)
        text = payload['thinking'].strip()
        if text and '<think' not in text and '</think' not in text:
            payload['thinking'] = '<think>' + text + '</think>'
    return payload


def valid_payload(payload):
    if not isinstance(payload, dict) or set(payload) != set(SCHEMA['required']):
        return False
    if not all(isinstance(v, str) for v in payload.values()):
        return False
    text = payload['thinking'].strip()
    # 객체 ID의 <...>는 허용하지만 think 중첩과 빈 설명은 허용하지 않습니다.
    return (text.startswith('<think>') and text.endswith('</think>')
            and bool(text[7:-8].strip()) and text.count('<think>') == text.count('</think>') == 1
            and not re.search(r"unable to generate thinking|I'm unable to assist|I cannot assist", text, re.I))


def export_results(run, samples, records, expected, details):
    # 3. 새 결과 JSON: 기존 필드 + thinking만 저장합니다. 토큰·비용은 별도 파일입니다.
    # 한쪽 실패 시 두 설정 모두 완료된 공통 QA만 내보내 항상 같은 데이터·같은 양을 유지합니다.
    matched = [s for s in samples if all(expected[(s['sample_id'], d)] in records for d in details)]
    for detail in details:
        result = []
        for sample in matched:
            record = records[expected[(sample['sample_id'], detail)]]
            generated = record['generated']
            row = {**sample, 'thinking': generated['thinking']}
            result.append(row)
        save_json(run / f'think_{detail}.json', result)
    comparison = [{'sample_id': sample['sample_id'], 'category': sample['category'],
                   'completed_details': details} for sample in matched]
    cost = sum(r.get('cost_usd', 0) for r in records.values())
    # 거절/길이 초과 응답도 과금될 수 있으므로 성공 비용과 전체 응답 비용을 구분합니다.
    invalid_path = run / 'invalid_responses.jsonl'
    invalid_rows = [json.loads(line) for line in invalid_path.read_text(encoding='utf-8').splitlines() if line.strip()] if invalid_path.exists() else []
    all_cost = sum(r.get('cost_usd', 0) for r in invalid_rows) + sum(
        r.get('cost_usd', 0) for r in records.values() if not r.get('recovered_from_invalid'))
    save_json(run / 'summary.json', {'selected_qa': len(samples), 'paired_qa': len(matched),
        'detail_modes': details, 'completed_responses_in_run': len(records),
        'recorded_success_cost_usd': round(cost, 6),
        'recorded_all_response_cost_usd': round(all_cost, 6),
        'cost_note': 'Standard uncached price estimate. Check provider billing for all failed/timeout requests.',
        'generation_note': 'Answer-conditioned explanation; not independent factual validation.',
        'comparison': comparison})
    return len(matched)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=30)
    parser.add_argument('--detail', choices=['both', 'low', 'high'], default='both')
    parser.add_argument('--model', default='gpt-4o-2024-08-06')
    parser.add_argument('--dry-run', action='store_true', help='축소·경로 검사만 하며 API 호출 없음')
    parser.add_argument('--input', type=Path, default=Path(__file__).resolve().parent / 'selected_30.json')
    parser.add_argument('--images', type=Path, default=Path(__file__).resolve().parent / 'select_30_image')
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit은 1 이상이어야 합니다.')
    base = Path(__file__).resolve().parent
    input_bytes = args.input.read_bytes()
    all_samples = json.loads(input_bytes)
    if not isinstance(all_samples, list) or not all_samples:
        raise ValueError('입력 JSON은 비어 있지 않은 배열이어야 합니다.')
    if len({s['sample_id'] for s in all_samples}) != len(all_samples):
        raise ValueError('중복 sample_id가 있습니다.')
    if args.limit > len(all_samples):
        parser.error(f'입력에 QA {len(all_samples)}개만 있습니다. --limit을 줄여주세요.')
    samples = all_samples[:args.limit]
    details = ['low', 'high'] if args.detail == 'both' else [args.detail]
    prepared = {}
    for sample in samples:
        if sample['category'] not in GUIDANCE or set(sample['image_paths']) != set(CAMERAS):
            raise ValueError(f"범주 또는 6방향 이미지 정보 오류: {sample['sample_id']}")
        infos, images = [], []
        for camera in CAMERAS:
            source = args.images / camera / Path(sample['image_paths'][camera]).name
            info, raw = resize_image(source, base / 'resized_512x288')
            info['camera'] = camera
            infos.append(info)
            images.append((camera, base64.b64encode(raw).decode('ascii')))
        prepared[sample['sample_id']] = (infos, images, build_prompt(sample, infos))

    # 4. 재사용 기준: sample_id뿐 아니라 실제 전송 이미지·원본 QA·프롬프트·설정을 해시합니다.
    settings = {'version': 'paired-v3-answer-conditioned', 'model': args.model, 'size': [512, 288],
                'temperature': 0, 'max_tokens': 700, 'system': SYSTEM,
                'guidance': GUIDANCE, 'schema': SCHEMA, 'input_sha256': digest(input_bytes)}
    expected, jobs = {}, {}
    for sample in samples:
        infos, images, prompt = prepared[sample['sample_id']]
        for detail in details:
            spec = {'settings': settings, 'sample': sample, 'images': infos, 'prompt': prompt, 'detail': detail}
            fp = digest(canonical(spec))
            expected[(sample['sample_id'], detail)] = fp
            jobs[fp] = spec
    run_hash = digest(canonical({'settings': settings,
        'requests': sorted(expected.values())}))[:16]
    run = base / 'gpt4o_comparison' / run_hash
    print(f'이미지 512x288 준비 완료: QA {len(samples)}개, 설정 {details}, 최대 {len(jobs)}회 호출')
    print(f'결과 폴더: {run}')
    if args.dry_run:
        print('실제 API 호출과 결과 생성은 하지 않았습니다. 원본 JSON/이미지는 보존했습니다.')
        return
    key = os.environ.get('OPENAI_API_KEY', '').strip()
    if not key or any(c.isspace() for c in key):
        raise RuntimeError('OPENAI_API_KEY에 공백 없는 API 키를 설정하세요.')
    from openai import OpenAI
    # 자동 재시도로 중복 비용이 생기지 않게 합니다. 실패 항목은 다음 실행에서 재시도합니다.
    client = OpenAI(api_key=key, timeout=180.0, max_retries=0)
    run.mkdir(parents=True, exist_ok=True)
    lock = run / '.running.lock'
    try:
        descriptor = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError('동일 실험이 실행 중입니다. 비정상 종료했다면 실행 프로세스 확인 후 .running.lock을 삭제하세요.') from None
    os.close(descriptor)
    failures = 0
    try:
        save_json(run / 'manifest.json', {'settings': settings, 'details': details,
            'sample_ids': [s['sample_id'] for s in samples], 'requests': jobs})
        ledger = run / 'responses.jsonl'
        records = {}
        if ledger.exists():
            for line in ledger.read_text(encoding='utf-8').splitlines():
                record = json.loads(line)
                fp = record['request_fingerprint']
                if fp not in jobs or not valid_payload(record.get('generated')):
                    raise ValueError('저장 응답의 입력/설정/형식이 현재 실험과 다릅니다.')
                records[fp] = record
        # 이전에 태그 누락만으로 제외한 응답을 비용 없이 복구합니다.
        invalid_file = run / 'invalid_responses.jsonl'
        if invalid_file.exists():
            for line in invalid_file.read_text(encoding='utf-8').splitlines():
                record = json.loads(line)
                fp = record['request_fingerprint']
                normalized = normalize_payload(record.get('generated'))
                if (fp in jobs and fp not in records and record.get('finish_reason') == 'stop'
                        and valid_payload(normalized)):
                    record = dict(record)
                    record['generated'] = normalized
                    record['recovered_from_invalid'] = True
                    append_record(ledger, record)
                    records[fp] = record
        export_results(run, samples, records, expected, details)
        for index, sample in enumerate(samples):
            infos, images, prompt = prepared[sample['sample_id']]
            # 실행 순서 편향을 줄이기 위해 샘플마다 low/high 순서를 교대합니다. 병렬 호출은 아닙니다.
            order = details if index % 2 == 0 else list(reversed(details))
            for detail in order:
                fp = expected[(sample['sample_id'], detail)]
                if fp in records:
                    print(f"재사용: {sample['sample_id']} / {detail}")
                    continue
                # 5. 이미지·프롬프트를 API에 전송: 두 설정은 같은 입력과 출력 조건을 사용합니다.
                content = [{'type': 'text', 'text': prompt}]
                for camera, encoded in images:
                    content.extend([{'type': 'text', 'text': f'Camera: {camera}'},
                        {'type': 'image_url', 'image_url': {
                            'url': f'data:image/jpeg;base64,{encoded}', 'detail': detail}}])
                print(f"API 호출: {sample['sample_id']} / {detail}", flush=True)
                try:
                    response = client.chat.completions.create(model=args.model,
                        messages=[{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': content}],
                        temperature=0, max_tokens=700,
                        response_format={'type': 'json_schema', 'json_schema': {
                            'name': 'driving_explanation', 'strict': True, 'schema': SCHEMA}})
                except Exception as error:
                    # 키나 요청 원문을 로그에 남기지 않습니다. 성공한 다른 쪽 응답은 유지합니다.
                    status = getattr(error, 'status_code', None)
                    append_record(run / 'errors.jsonl', {'sample_id': sample['sample_id'], 'detail': detail,
                        'error_type': type(error).__name__, 'http_status': status})
                    if status in (400, 401, 403, 404, 429) or status is None:
                        raise RuntimeError(f'API 요청 실패 ({type(error).__name__}, HTTP {status}). 설정/키/잔액/한도를 확인하세요. 저장 결과는 유지됩니다.') from None
                    failures += 1
                    continue
                usage = response.usage.model_dump() if response.usage else {}
                prompt_tokens = usage.get('prompt_tokens', 0)
                completion_tokens = usage.get('completion_tokens', 0)
                cost = prompt_tokens * 2.5 / 1_000_000 + completion_tokens * 10 / 1_000_000
                choice = response.choices[0] if response.choices else None
                text = (choice.message.content or '') if choice else ''
                try:
                    generated = normalize_payload(json.loads(text))
                except (ValueError, TypeError):
                    generated = None
                # 6. 생성된 설명과 사용량을 기록합니다. 정답 타당성 판정은 수행하지 않습니다.
                record = {'sample_id': sample['sample_id'], 'detail': detail,
                    'request_fingerprint': fp, 'model_returned': response.model,
                    'generated_at_utc': datetime.now(timezone.utc).isoformat(),
                    'usage': usage, 'cost_usd': cost, 'generated': generated,
                    'finish_reason': choice.finish_reason if choice else None}
                if choice is None or choice.finish_reason != 'stop' or not valid_payload(generated):
                    record['raw_text'] = text
                    append_record(run / 'invalid_responses.jsonl', record)
                    failures += 1
                    print('응답 형식/거절/출력 길이 오류: 별도 저장, 다음 실행에서 재시도 가능')
                    continue
                # 7. 응답을 먼저 보존하고, 두 설정 공통 샘플만 새 JSON에 내보냅니다.
                append_record(ledger, record)
                records[fp] = record
                paired = export_results(run, samples, records, expected, details)
                print(f"저장: {sample['sample_id']} / {detail} / ${cost:.5f}; 공통 QA {paired}개")
        paired = export_results(run, samples, records, expected, details)
        print(f'완료: 설정별 공통 QA {paired}/{len(samples)}개. 결과: {run}')
        if failures:
            print('실패 항목이 남았습니다. 같은 명령으로 재실행하면 미완료 응답만 호출합니다.')
            return 1
        return 0
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    sys.exit(main())

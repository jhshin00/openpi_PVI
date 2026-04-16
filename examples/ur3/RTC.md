# UR3 RTC Integration Notes

이 문서는 `examples/ur3`에 추가한 Real-Time Chunking(RTC) 구현이 실제로 어떻게 동작하는지 설명한다.

대상은 다음 파일들이다.

- `examples/ur3/main_RTC.py`
- `examples/ur3/main_RTC_interactive.py`
- `src/openpi/models_pytorch/pi0_pytorch.py`
- `src/openpi/models_pytorch/pi0_pvi_pytorch.py`

중요한 전제:

- 여기서 사용한 것은 inference-time RTC다.
- upstream `real-time-chunking-kinetix` README에 있는 training-time action conditioning(`simulated_delay`로 fine-tune) 경로는 사용하지 않는다.
- 현재 구현은 UR3에서 쓰는 PyTorch `pi0/pi0.5` 계열, 특히 `pi0.5 + PVI` 체크포인트를 기준으로 붙였다.

## 1. 왜 기존 async chunking만으로는 부족한가

기존 `main_modify.py`의 async inference는 대략 이렇게 동작한다.

1. 현재 observation으로 policy chunk를 요청한다.
2. 결과가 늦게 오면 `delay_steps`만큼 앞부분을 잘라낸다.
3. 남은 액션을 실행 plan으로 바꿔서 쓴다.

이 방식은 "늦게 도착한 chunk의 앞부분을 버린다"는 점에서는 latency를 부분적으로 처리하지만, 새 chunk 자체는 이전 chunk를 조건으로 생성되지 않는다.

즉:

- 이미 실행되었거나 곧 실행될 prefix와 새 chunk의 prefix가 자연스럽게 이어진다는 보장이 없다.
- chunk boundary에서 멈칫하거나 jerk가 생길 수 있다.

RTC는 이 부분을 해결하려고, 새 chunk를 만들 때 이전 chunk의 prefix를 soft/hard하게 맞추도록 유도한다.

## 2. 이번 구현의 핵심 아이디어

UR3 쪽 구현은 RTC를 두 레이어로 나눠 붙였다.

1. 모델 레이어
   - `sample_actions()` 외에 `realtime_action()`을 추가했다.
   - 이 함수는 `prev_action_chunk`를 조건으로 받아, 이전 chunk와 이어지는 새 chunk를 샘플링한다.

2. 실행 스케줄러 레이어
   - `main_RTC.py`의 `_RTCPlanner`가 "현재 실행 중인 chunk"와 "다음 chunk"를 관리한다.
   - `replan_steps`를 paper의 `execute_horizon`처럼 사용한다.
   - 한 iteration 동안 `replan_steps`개를 실행하는 동안, background thread에서 다음 chunk를 미리 만든다.

정리하면:

- 모델은 "이전 chunk와 이어지는 새 chunk"를 만든다.
- 스케줄러는 "현재 chunk 일부를 실행하면서, 새 chunk가 도착하면 남은 구간과 다음 iteration에 반영"한다.

## 3. main_RTC.py 구조

`main_RTC.py`는 `main_modify.py`를 완전히 다시 쓰지 않고, 기존 helper를 최대한 재사용한다.

- `_base = _load_base_module()`로 `main_modify.py`를 backend처럼 로드한다.
- env 생성, video 처리, observation 변환, chunk endpoint interpolation 같은 기존 helper는 그대로 가져다 쓴다.
- RTC에 필요한 것만 새로 추가했다.

새로 들어간 주요 구성요소는 아래와 같다.

### `Args`

기존 `Args`에 RTC 파라미터를 추가했다.

- `rtc_inference_delay_steps`
  - RTC가 처음 사용할 delay step 값
  - dynamic delay가 켜져 있으면 초기값이자 최소값처럼 동작한다.
- `rtc_dynamic_inference_delay`
  - 최근 실제 inference latency를 바탕으로 `inference_delay`를 자동 갱신할지 여부
- `rtc_delay_history`
  - dynamic delay 추정에 사용할 최근 observed delay history 길이
- `rtc_prefix_attention_schedule`
  - prefix matching weight schedule
  - `linear`, `exp`, `ones`, `zeros`
- `rtc_max_guidance_weight`
  - RTC correction 세기의 상한

또한 RTC 모드에서는:

- `async_inference=True`가 강제된다.
- `replan_steps >= rtc_inference_delay_steps`여야 한다.
- `replan_steps <= action_horizon`이어야 한다.

## 4. `_RTCPolicyAdapter`: policy와 RTC sampler를 연결하는 층

`_RTCPolicyAdapter`는 openpi의 일반 `Policy` wrapper와 실제 PyTorch model 사이를 이어준다.

이 adapter가 하는 일은 2가지다.

1. 초기 chunk 생성
   - `infer_initial()`
   - 기존 `sample_actions()`를 이용해 첫 chunk를 만든다.

2. RTC chunk 생성
   - `infer_realtime()`
   - `model.realtime_action(...)`을 호출해서 이전 chunk와 이어지는 새 chunk를 만든다.

여기서 중요한 점:

- `model_actions`
  - output transform 이전의 model-space action chunk
  - RTC conditioning은 이것을 사용한다.
- `actions`
  - output transform 이후 실제 UR3에 보낼 action chunk
  - 실행은 이것을 사용한다.

즉 RTC는 "모델이 실제로 생성한 latent/action space chunk"를 기준으로 연결성을 맞추고, UR3 실행은 기존 output transform을 거친 결과를 그대로 따른다.

## 5. 모델 쪽 RTC 구현

### 5.1 공통 함수

`src/openpi/models_pytorch/pi0_pytorch.py`에 아래 helper를 추가했다.

- `get_prefix_weights(...)`
  - upstream RTC repo의 prefix weighting schedule을 PyTorch로 옮긴 것
- `get_rtc_guidance_weight(...)`
  - openpi의 reverse-time flow parameterization에 맞게 guidance weight를 계산하는 함수

### 5.2 `PI0Pytorch.realtime_action()`

기본 PI0/Pi0.5 PyTorch 모델에는 `realtime_action()`을 추가했다.

흐름은 다음과 같다.

1. observation을 기존 `sample_actions()`와 동일하게 preprocess한다.
2. prefix(image/language) KV cache를 한 번 만든다.
3. 초기 noise `x_t`에서 시작해 `t=1 -> 0`으로 Euler 적분한다.
4. 각 step에서:
   - `denoise_step(...)`으로 현재 velocity `v_t`를 계산한다.
   - 현재 denoised action estimate를 `action_estimate = x_t - t * v_t`로 만든다.
   - `prev_action_chunk`와의 차이를 prefix weight로 가중한다.
   - `autograd.grad`로 `d(action_estimate)/d(x_t)`를 따라 pseudo-inverse style correction을 얻는다.
   - `corrected_velocity = v_t + guidance_weight * pinv_correction`으로 보정한다.
5. 보정된 velocity로 다음 `x_t`를 적분한다.
6. 마지막 `x_t`를 새 RTC chunk로 반환한다.

핵심은 "이전 chunk의 앞부분과 잘 맞는 방향으로 denoising trajectory를 유도"하는 것이다.

### 5.3 `PI0PVI.realtime_action()`

`pi0.5 + PVI`는 sampling path가 기본 PI0와 다르기 때문에 `src/openpi/models_pytorch/pi0_pvi_pytorch.py`에도 별도로 RTC를 넣었다.

추가된 핵심 함수:

- `_pvi_denoise_step(...)`
  - PVI의 main prefix hidden state + auxiliary prefix + copy expert 경로를 사용해 현재 step의 velocity를 계산
- `realtime_action(...)`
  - 위 velocity를 기반으로 동일한 RTC correction loop 수행

즉 PVI에서도 RTC는 같은 원리로 동작하지만, denoiser 자체는 기존 PVI sampling path를 그대로 사용한다.

## 6. `_RTCPlanner`: 실행 스케줄링

paper의 중요한 포인트는 "새 chunk를 생성하는 동안 현재 chunk를 계속 실행"하는 것이다.

UR3에서는 `_RTCPlanner`가 이 역할을 한다.

planner가 관리하는 상태는 대략 아래와 같다.

- `current_model_chunk`
  - 현재 iteration의 기준이 되는 model-space chunk
- `current_env_chunk`
  - 실제 실행용 env-space chunk
- `next_model_chunk`
  - 다음 iteration에 쓸 다음 model-space chunk
- `next_env_chunk`
  - 다음 iteration에 쓸 다음 env-space chunk
- `action_plan`
  - 현재 step에서 당장 꺼내 실행할 deque

### bootstrap

episode 시작 시:

1. `infer_initial()`로 첫 full chunk를 만든다.
2. 이 chunk의 앞 `replan_steps`개를 현재 iteration 실행 plan으로 만든다.
3. 동시에 background worker에 "다음 chunk를 RTC로 만들어라"는 요청을 보낸다.

### 한 iteration 동안

iteration은 `replan_steps` step 길이다.

iteration 중에는:

1. `action_plan`에서 하나씩 액션을 꺼내 실행한다.
2. background worker가 RTC chunk를 계산한다.
3. 결과가 도착하면:
   - `actual_delay = step - request_step`를 계산한다.
   - `timing_delay = ceil(infer_ms / control_period)`를 계산한다.
   - `observed_delay = max(actual_delay, timing_delay)`를 최근 history에 넣는다.
   - 이미 지나간 구간은 버린다.
   - 남은 `result.chunk.actions[actual_delay : replan_steps]`를 현재 observation 기준으로 다시 실행 plan으로 바꿔 넣는다.
   - 동시에 `next_model_chunk`, `next_env_chunk`를 다음 iteration용으로 준비한다.

여기서 `chunk_execution`은 여전히 살아 있다.

- `per_step`: 모델 chunk를 그대로 step-by-step 실행
- `chunk_endpoint`: 앞 `replan_steps` 구간을 endpoint interpolation으로 execution plan으로 변환

즉 RTC는 "chunk 생성 방식"을 바꾸고, execution shaping은 기존 UR3 로직을 그대로 재사용한다.

### iteration 경계

`replan_steps`개를 다 실행하면:

1. 새 chunk가 준비되어 있으면
   - `current_* = next_*`로 교체
2. 아직 준비되지 않았으면
   - 현재 chunk를 `replan_steps`만큼 left shift해서 fallback으로 사용

shift는 `_shift_chunk()`가 처리한다.

- 앞 `shift`개를 버린다.
- 모자라는 뒤쪽은 마지막 action을 반복해서 채운다.

이 부분은 paper의

- 현재 horizon을 실행하고
- 새 chunk를 다음 frame of reference에 맞게 shift해서 이어 쓰는

구조에 대응한다.

### dynamic inference delay

현재 구현은 LeRobot 스타일을 일부 가져와서, request마다 모델에 넣는 `inference_delay`를 고정값으로 두지 않을 수 있다.

동작 방식:

1. 첫 요청은 `rtc_inference_delay_steps`를 사용한다.
2. 각 RTC 결과가 도착하면
   - `actual_delay`
   - `timing_delay = ceil(infer_ms / control_period)`
   - `observed_delay = max(actual_delay, timing_delay)`
   를 계산한다.
3. 최근 `rtc_delay_history`개 observed delay의 최대값을 다음 request의 `inference_delay`로 사용한다.
4. 다만 scheduler의 가정(`execute_horizon >= inference_delay`)을 깨지 않도록, 실제 모델에 넣는 값은 `replan_steps`로 clamp한다.

즉:

- `rtc_inference_delay_steps`는 이제 "고정 delay"라기보다 "초기값/최소값"
- `request_delay`는 실제 해당 inference 호출에 모델이 사용한 delay
- `observed_delay`는 이전 호출들에서 측정된 실제 지연

으로 이해하면 된다.

## 7. 비동기 worker 구조

`_start_inference_worker()`는 queue 크기 1짜리 worker thread를 띄운다.

특징:

- request도 1개만 유지
- result도 1개만 유지
- 새 request/result를 넣기 전에 queue를 drain

즉 여러 개를 파이프라인으로 쌓는 구조가 아니라, 항상 "가장 최신 observation 기준의 다음 RTC chunk 1개"만 유지한다.

stale result는 `_RTCPlanner.maybe_refresh()`에서 iteration index로 걸러진다.

## 8. `main_modify.py`와의 차이

기존 `main_modify.py` async mode는:

- observation이 들어오면 policy chunk를 요청
- 도착 시 `delay_steps`만큼 잘라서 plan 교체

새 `main_RTC.py`는:

- 첫 chunk는 일반 sampling
- 이후 chunk는 항상 `prev_model_actions`를 조건으로 한 `realtime_action()`으로 생성
- `replan_steps`를 고정 execution horizon처럼 사용
- iteration 단위로 current/next chunk를 관리

즉 차이의 본질은 "stale-trim"에서 "conditioned next-chunk generation + horizon scheduler"로 바뀌었다는 점이다.

## 9. `main_RTC_interactive.py`

interactive GUI는 새로 처음부터 쓰지 않고 기존 `main_interactive.py`를 재사용했다.

전략은 아래와 같다.

1. 기존 GUI 모듈을 `_gui`로 로드
2. RTC backend인 `main_RTC.py`를 `_backend`로 로드
3. `_gui._backend = _backend`로 교체
4. `InteractivePolicySession`을 subclass해서 아래만 override
   - `_initialize()`
   - `_start_inference_worker_if_needed()`
   - `_run_episode()`

그래서 아래 요소는 기존 GUI 구현을 그대로 쓴다.

- Flask UI
- preview image
- video save/discard flow
- 상태 snapshot API

반대로 episode control의 핵심 loop만 RTC planner로 바뀐다.

## 10. 파라미터 의미

실제로는 아래처럼 이해하면 된다.

- `replan_steps`
  - RTC의 `execute_horizon`
  - 한 iteration에서 실제로 실행할 step 수
- `rtc_inference_delay_steps`
  - dynamic delay의 초기값
  - `rtc_dynamic_inference_delay=False`면 고정 delay로 동작
- `rtc_dynamic_inference_delay`
  - 최근 실제 latency를 이용해 request별 `inference_delay`를 자동 갱신할지 여부
- `rtc_delay_history`
  - 최근 몇 개 result의 observed delay를 보고 다음 delay를 추정할지
- `rtc_prefix_attention_schedule`
  - prefix mismatch를 얼마나 강하게 볼지 정하는 schedule
- `rtc_max_guidance_weight`
  - RTC correction strength 상한

현재 구현에서 명시적으로 무시되는 값:

- `async_prefetch_steps`
- `async_plan_guard_steps`

로그에도 `rtc_scheduler=enabled ... ignored=...` 형태로 남기도록 했다.

## 11. 로그에서 볼 것

RTC가 제대로 붙었는지 보려면 아래 로그를 보면 된다.

- `rtc_bootstrap`
  - 첫 chunk 생성 시간과 horizon 정보
- `rtc_iteration`
  - 현재 iteration에서 실제 실행 plan이 어떻게 생겼는지
- `rtc_update`
  - 결과가 언제 도착했고, 모델에 넣은 delay와 실제 delay가 어떻게 달랐는지
- `rtc_fallback`
  - 다음 chunk가 늦어서 현재 chunk를 shift 재사용했는지

특히 `rtc_update`의 값이 중요하다.

- `request_delay`
  - 이번 inference 호출에서 실제로 모델에 넣은 delay
- `actual_delay`
  - 실제로 step 기준 몇 step 늦게 도착했는지
- `timing_delay`
  - `infer_ms / control_period`로 환산한 대략적인 지연 step
- `observed_delay`
  - `max(actual_delay, timing_delay)`
- `next_delay_estimate`
  - 다음 request에 쓸 예정인 clamped delay
- `raw_next_delay_estimate`
  - clamp 전 delay 추정값

실험할 때는 `request_delay`와 `observed_delay`가 계속 벌어지는지, 그리고 `rtc_delay_exceeds_horizon` 경고가 뜨는지를 보는 것이 좋다.

## 12. 현재 구현의 제한

현재 문맥에서 의도적으로 제한한 부분이 있다.

- PyTorch policy만 지원
  - adapter는 `policy._is_pytorch_model`과 `model.realtime_action` 존재를 확인한다.
- training-time RTC는 지원하지 않음
  - `simulated_delay` fine-tuning 경로는 포함하지 않았다.
- RTC conditioning은 model-space action chunk 기준
  - env-space가 아니라 output transform 이전 chunk를 조건으로 쓴다.
- execution shaping은 기존 UR3 방식 유지
  - 따라서 paper의 action execution과 완전히 동일한 low-level control은 아니다.

하지만 현재 UR3 evaluation 코드와 가장 적게 충돌하면서 inference-time RTC의 핵심만 가져오는 방향으로는 이 구조가 가장 안전하다.

## 13. 실전 사용 팁

초기 권장값:

- `replan_steps=8`
- `rtc_inference_delay_steps=2` 또는 `3`
- `rtc_dynamic_inference_delay=true`
- `rtc_delay_history=8`
- `rtc_prefix_attention_schedule=exp`
- `rtc_max_guidance_weight=5.0`

만약:

- 결과가 늦게 와서 `rtc_fallback`이 자주 뜨면
  - `replan_steps`를 조금 늘리거나
  - 실제 inference latency를 줄여야 한다.
- chunk 연결은 좋아졌는데 과하게 끌려가는 느낌이 있으면
  - `rtc_max_guidance_weight`를 낮춰볼 수 있다.
- 실제 지연이 더 큰데 mismatch가 보이면
  - `rtc_inference_delay_steps`를 올려서 초기값 자체를 보수적으로 잡을 수 있다.
  - 또는 `rtc_delay_history`를 늘려 더 보수적으로 추정할 수 있다.

## 14. 한 줄 요약

이번 UR3 RTC 구현은:

- 모델 쪽에서는 `prev_action_chunk`를 조건으로 새 chunk를 생성하는 `realtime_action()`을 추가했고
- 실행 쪽에서는 `replan_steps`를 execution horizon으로 두는 `_RTCPlanner`를 추가해
- 기존 async stale-trim을 "실시간 chunk stitching" 방식으로 바꾼 것이다.

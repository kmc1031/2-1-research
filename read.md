# x264 + 3D DT-CWT 전처리 연구 보완 실험 계획

## 1. 실험 목적

현재 연구의 핵심 목표는 다음과 같다.

> **x264 인코딩 전에 3D DT-CWT 전처리를 적용했을 때, 저비트레이트 환경에서 영상 품질(PSNR, VMAF 등)을 개선할 수 있는가?**

다만 현재 공개된 결과를 보면,  
- **clean 영상**에서는 proposed method가 baseline보다 불리한 경우가 많고  
- **noisy 영상**에서는 일부 bitrate 구간에서만 제한적 개선이 보인다.

따라서 다음 보완 실험의 목적은  
**“항상 개선된다”를 증명하는 것**이 아니라,  
**“어떤 조건에서, 왜 개선되는가”를 정교하게 규명하는 것**이다.

---

## 2. 보완 실험 전체 계획표

| 우선순위 | 실험 이름 | 핵심 질문 | 독립변수 | 통제조건 | 측정 지표 | 성공 판정 | 실패 시 해석 / 다음 액션 |
|---|---|---|---|---|---|---|---|
| 1 | 도메인 한정 실험: clean vs noisy | DT-CWT 전처리가 **노이즈가 있는 저비트레이트 조건**에서만 유효한가? | 입력 조건: clean / Gaussian noise(σ=5,10,15) / compression noise / mixed noise | 영상, bitrate, x264 설정, threshold, chunk 크기 동일 | PSNR, SSIM, VMAF, MS-SSIM, EPSNR, PSNR-B, GBIM, MEPR | noisy 조건에서만 BD-rate < 0 또는 같은 bitrate에서 VMAF/PSNR 일관 개선 | clean에서 계속 지면 “범용 전처리” 주장을 버리고 **noisy low-bitrate 특화 전처리**로 재정의 |
| 2 | 인코더 설정 분리 실험 | 성능 차이가 전처리 효과인지, 아니면 특정 x264 설정에만 유효한지? | x264 setting: `fast+zerolatency` / `medium` / `slow`, 그리고 CRF vs bitrate 모드 | 입력 영상과 전처리 파라미터 동일 | RD curve, BD-rate(PSNR 기준), BD-rate(VMAF 기준), 인코딩 시간 | 특정 setting에서만 개선되면 적용 범위를 그 조건으로 명시 가능 | 모든 setting에서 baseline보다 열세면 threshold/scale 재탐색 필요 |
| 3 | 전처리 강도 ablation | threshold와 bitrate-dependent scaling이 어느 값에서 최적인가? | threshold 3~5개 값, scaling on/off, lowpass만 처리 vs directional subband 포함 | 영상, encoder setting, noise level 고정 | VMAF, PSNR-B, GBIM, 시각 비교 프레임 | U자형 최적점이 보이면 “과전처리-과소전처리 tradeoff” 설명 가능 | 단조 하락이면 현재 전처리 방식이 과도함 → subband별 선택적 억제로 수정 |
| 4 | 시간축 효과 검증 | 진짜 성능 이득이 “3D 처리”에서 오는가? | 3D DT-CWT / framewise 2D DT-CWT / Gaussian blur / 3D DWT baseline | encoder, bitrate, noise 동일 | VMAF, MS-SSIM, MEPR, temporal artifact 관찰 | 3D가 2D보다 MEPR·VMAF에서 우세하면 시간축 기여 주장 가능 | 차이가 없으면 3D 복잡도 정당화 어려움 → 2D 간소화 버전 고려 |
| 5 | chroma 처리 ablation | Y 채널 이득인지, U/V 처리까지 하는 것이 유리한지? | Y only / Y+UV / UV off / UV 약한 threshold | 나머지 동일 | PSNR(Y), VMAF, 색번짐 시각 평가 | Y only가 더 좋으면 chroma 전처리 과도 가능성 확인 | 색 손상이 보이면 UV는 별도 파라미터로 분리 |
| 6 | chunk / overlap 민감도 | overlap-save와 chunk 길이가 경계 artifact나 temporal consistency에 어떤 영향을 주는가? | chunk 길이 T=8,16,32,64 / overlap=0,25%,50% | 영상, noise, threshold 동일 | MEPR, VMAF, 경계 프레임 시각 비교, 처리속도 | overlap 증가로 temporal 품질이 개선되면 현재 경계 처리 방식의 타당성 확보 | 차이 없으면 구현 단순화 가능 |
| 7 | content class 확장 | 효과가 특정 영상 유형에만 나타나는가? | 저움직임 / 중간움직임 / 고움직임 / 텍스처 풍부 / 애니메이션 등 4~6개 클래스 | bitrate set, encoder, noise protocol 동일 | 영상별 BD-rate, 평균·분산, boxplot | 특정 class에서만 개선되면 그 특성을 핵심 결과로 제시 가능 | 영상마다 결과가 들쑥날쑥하면 scene-adaptive gating 필요 |
| 8 | scene-adaptive on/off gating | 모든 프레임에 적용하지 않고 noisy/복잡한 구간에만 적용하면 더 좋아지는가? | always-on / scene-adaptive / frame-adaptive | 동일 bitrate, 동일 영상 | 전체 VMAF, 구간별 VMAF, PSNR-B, 처리시간 | always-on보다 adaptive가 낫다면 “조건부 전처리”가 핵심 공헌 | 구간 판단이 불안정하면 단순 classifier로 축소 |
| 9 | 속도-품질 tradeoff | GPU DT-CWT 비용이 실용성을 해치지 않는가? | CPU vs GPU / chunk size / batch size | 동일 품질 설정 | fps, sec/frame, GPU memory, 품질 저하량 | 품질 유지하며 처리시간이 실용 범위면 시스템 기여 강화 | 너무 느리면 “오프라인 전처리” 시나리오로 적용 범위 축소 |
| 10 | 지표 정합성 정리 실험 | 결론이 특정 지표에만 의존하는가? | PSNR / VMAF / MS-SSIM / EPSNR / PSNR-B / GBIM / MEPR 전체 재수집 | 영상·bitrate 동일 | 지표 간 상관, 순위 일치도 | 주요 결론이 여러 지표에서 재현되면 신뢰성 상승 | 지표 충돌 시 “blocking 개선 vs detail 손실” 식으로 해석 분리 |

---

## 3. 가장 먼저 해야 할 1차 핵심 실험 세트

아래 4개는 **논문/발표 설득력을 가장 빠르게 올리는 핵심 실험**이다.

| 1차 핵심 세트 | 왜 먼저 해야 하는가 | 최소 실험 설계 |
|---|---|---|
| 도메인 한정 실험 | 현재 결과가 clean에서는 약하고 noisy에서만 일부 가능성이 보이기 때문 | `foreman + 3개 영상`, `clean / σ=5 / σ=10 / σ=15`, bitrate `100/200/300/400/500 kbps` |
| 인코더 설정 분리 | 현재 파이프라인이 `fast+zerolatency`라 일반적인 코덱 효율 개선으로 일반화하기 어렵기 때문 | `fast+zerolatency` vs `medium` vs `slow`, 영상 3개, bitrate 3개 |
| 전처리 강도 ablation | 현재 성능 저하의 가장 유력한 원인이 과전처리이기 때문 | threshold 5점, scaling on/off, noisy σ=10 기준 |
| 3D vs 2D 비교 | “왜 굳이 3D DT-CWT인가?”를 증명해야 하기 때문 | `3D DT-CWT / 2D DT-CWT / Gaussian / DWT / raw baseline` 비교 |

---

## 4. 단계별 실행 계획

| 단계 | 해야 할 일 | 산출물 |
|---|---|---|
| Phase A | 평가 코드 정리, STRRED/MEPR 명칭 혼선 수정, 실패값 처리 방식 정리 | 신뢰 가능한 evaluation 스크립트 |
| Phase B | clean/noisy + encoder setting + threshold ablation 수행 | 핵심 RD curve, BD-rate 표 |
| Phase C | 3D vs 2D vs Gaussian vs DWT 비교, content class 확장 | 공헌점 설명용 메인 결과 그림 |
| Phase D | adaptive gating, speed profiling, qualitative frame 정리 | 발표용 시스템 다이어그램 + qualitative figure |

---

## 5. 권장 연구 가설

현재 결과를 바탕으로 가장 설득력 있게 세울 수 있는 가설은 다음과 같다.

| 가설 | 내용 |
|---|---|
| H1 | 3D DT-CWT 전처리는 **모든 영상**이 아니라 **noisy low-bitrate 조건**에서 더 효과적이다 |
| H2 | 그 효과는 단순 blur가 아니라 **방향성 보존 + 시간축 일관성 유지**에서 온다 |
| H3 | 항상-on 전처리보다 **scene-adaptive 전처리**가 RD 성능을 더 안정적으로 만든다 |

---

## 6. 보고서/발표에서 바로 쓸 수 있는 정리

| 항목 | 권장 문장 |
|---|---|
| 연구 목적 | “DT-CWT 전처리가 모든 영상에서 x264를 개선하는가?”가 아니라, “어떤 조건에서 개선되는가?”를 규명 |
| 핵심 비교군 | raw x264 / Gaussian / 3D DWT / 2D DT-CWT / 3D DT-CWT |
| 핵심 성공 기준 | noisy 조건 평균 BD-rate < 0, clean 조건 손실 최소화, MEPR·PSNR-B 동반 개선 |
| 핵심 실패 기준 | clean/noisy 모두에서 baseline보다 지속적 열세 |
| 실패 시 연구 전환 | 범용 전처리 → **조건부 전처리 또는 noisy-domain 특화 연구**로 전환 |

---

## 7. 최종 권장 실행 순서

현재 레포 상태를 기준으로, 가장 추천하는 순서는 다음과 같다.

1. **평가 코드 정리**
   - 지표 이름 정합성 수정
   - 실패값 처리 방식 수정
   - BD-rate 계산 안정화

2. **clean / noisy × encoder setting 실험**
   - 전처리 효과의 적용 범위 확인
   - 특정 인코더 setting 의존성 확인

3. **threshold / scaling ablation**
   - 과전처리 여부 확인
   - 최적 강도 탐색

4. **3D vs 2D / Gaussian / DWT 비교**
   - 3D DT-CWT의 필요성 입증
   - 단순 smoothing과의 차별성 확보

---

## 8. 핵심 결론

이 연구의 다음 단계는  
**“DT-CWT 전처리가 x264를 항상 개선한다”는 주장을 강화하는 것**이 아니라,  
다음 질문에 답하는 방향으로 가는 것이 가장 바람직하다.

- 어떤 입력 조건에서 효과가 있는가?
- 효과가 있다면 그 원인은 무엇인가?
- 단순 blur나 DWT보다 왜 DT-CWT가 나은가?
- 항상 적용하는 것보다 조건부 적용이 더 좋은가?

즉, 보완 실험의 핵심은 **범용성 증명**이 아니라  
**적용 조건과 작동 메커니즘의 정밀한 규명**이다.
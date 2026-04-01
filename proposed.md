2-1-research 저장소 개선 요청서
목적

kmc1031/2-1-research 저장소를
재현 가능한 연구 저장소이자 설득력 있는 연구 결과물 수준으로 개선하고자 한다.

현재 저장소는 3D DT-CWT 기반 비디오 전처리, x264 인코딩 비교, RD curve/BD-rate 평가, GPU 경로 등을 포함하고 있어 연구 아이디어와 구현 범위는 충분히 좋다. 다만 공개 연구 저장소 기준으로는 문서화, 재현성, 비교 실험, 구조화, 라이선스 측면에서 부족한 점이 있다.

현재 판단
강점
3D DT-CWT 기반 전처리 아이디어가 분명하다.
CPU/자체 GPU 구현 경로가 존재한다.
RD curve, BD-rate, PSNR, SSIM, VMAF, MS-SSIM 등 평가 파이프라인이 비교적 잘 갖춰져 있다.
chunk 처리, overlap, caching 등 실용적인 엔지니어링 요소가 포함되어 있다.
핵심 문제
재현 가이드가 부족하다.
README와 실제 구조가 완전히 일치하지 않는다.
성능 개선 주장이 모든 조건에서 일관되게 입증되지는 않는다.
비교 baseline이 더 강해져야 한다.
테스트/CI/패키징/배포 체계가 약하다.
LICENSE가 없다.
에이전트가 우선적으로 해야 할 일
1. 재현성 강화

다음을 README 또는 별도 REPRODUCIBILITY.md에 정리해라.

Python 버전
uv sync 기반 설치 절차
FFmpeg 요구사항
libx264
libvmaf
CUDA 사용 조건
최소 실행 예제 1개
기대 출력 파일 목록
결과 검증 방법

중요: 이 저장소는 Python 의존성만 맞춘다고 바로 재현되지 않는다. FFmpeg의 libvmaf 포함 여부가 핵심이다.

2. 연구 주장 범위 조정

현재 저장소의 메시지를
“전반적으로 품질이 좋아진다”에서
“특정 조건, 특히 noisy + low bitrate 환경에서 효과가 있을 가능성을 분석한다”로 더 정확하게 조정해라.

이유:
예시 결과에서는 낮은 bitrate 구간에서는 개선이 보이지만, 더 높은 bitrate에서는 baseline보다 VMAF나 PSNR이 낮은 구간도 있다. 따라서 현재 결과는 “항상 우수”보다는 “조건부 이점”에 가깝다.

3. baseline 비교 강화

다음 비교군을 추가해라.

x264 기본 baseline
x264 --nr 사용 baseline
FFmpeg 표준 denoise filter baseline
예: hqdn3d
필요시 nlmeans
현재 구현된 Gaussian / DWT3D baseline
가능하다면 최신 learned preprocessing 계열과 정성 비교

목적:
“왜 DT-CWT 기반 방식이 필요한가”를 더 설득력 있게 보여주기 위함이다.

4. 결과 정리 방식 개선

README에 다음 요약 표를 추가해라.

시퀀스별 평균 PSNR
시퀀스별 평균 VMAF
bitrate별 baseline 대비 증감
평균 BD-rate
clean / noisy 분리 결과
평균과 표준편차

또한 결과 해석은 반드시
“어떤 조건에서 개선되고, 어떤 조건에서 손해가 나는지”
형태로 작성해라.

5. 코드 구조 정리

현재 스크립트 중심 구조를 다음처럼 정리해라.

src/ : 핵심 모듈
scripts/ : 실행 스크립트
tests/ : 테스트
설정/상수 파일 분리
CLI entrypoint 제공

목표:
연구용 스크립트 모음이 아니라, 구조가 분명한 프로젝트처럼 보이게 만드는 것. README의 구조 설명과 실제 파일 배치도 일치시켜라.

6. 테스트 및 CI 추가

최소한 아래 테스트를 자동화해라.

forward → inverse reconstruction consistency
CPU / GPU 결과 차이 허용 범위 검증
chunk overlap / caching consistency
짧은 샘플 비디오 smoke test
metric 계산 정상 동작 확인

가능하면 GitHub Actions도 추가해라. 현재는 테스트 파일은 있으나 자동화 체계는 약하다.

7. 라이선스 추가

반드시 LICENSE 파일을 추가해라.

권장:

MIT
Apache-2.0

이유:
현재는 공개 저장소이지만 라이선스가 없어 외부 재사용이 법적으로 불명확하다.

우선순위
최우선
LICENSE 추가
재현 가이드 작성
최소 실행 예제 + 기대 결과 명시
다음 단계
결과 요약 표 정리
baseline 강화
구조 정리 (src/, scripts/, tests/)
장기 과제
CI 추가
여러 시퀀스/노이즈 조건 통계화
논문형 포지셔닝 보완
산출물 요구사항

에이전트는 최종적으로 아래 결과를 제공해야 한다.

개선된 README 초안
REPRODUCIBILITY.md 초안
필요한 디렉터리 재구성안
추가할 baseline 실험 계획
테스트 항목 목록
LICENSE 추천안
결과 요약 표 템플릿
최종 목표

이 저장소를
“좋은 연구 코드”에서
“남이 실행하고 검증하고 인용할 수 있는 연구 저장소”
수준으로 끌어올리는 것이 목표다.
핵심은 기능 추가보다 재현성, 비교 실험, 문서화, 구조화를 먼저 강화하는 것이다.
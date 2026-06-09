# Unreal Engine 머신러닝/AI 완벽 가이드

> **최신 버전 기준**: Unreal Engine 5.6 (2026년)  
> **주요 툴킷**: UnrealMLAgents v0.0.9, Schola v2.0, UE5 Learning Agents (Experimental)  
> **Python**: 3.10 ~ 3.12, PyTorch 2.x, Stable-Baselines3  
> **환경 관리**: venv (Python 내장) 또는 Conda (Miniconda 권장)  
> **참고 문서**: https://dev.epicgames.com/documentation/

---

## 목차

1. [Unreal Engine ML 개요](#1-unreal-engine-ml-개요)
2. [주요 ML 툴킷 비교](#2-주요-ml-툴킷-비교)
3. [환경 설정 (Environment Setup)](#3-환경-설정-environment-setup)
4. [커리큘럼](#4-커리큘럼)
   - 4.1 [Dodge 프로젝트 (UnrealMLAgents)](#41-dodge-프로젝트-unrealmlagents)
   - 4.2 [PPO 알고리즘](#42-ppo-알고리즘)
   - 4.3 [Attention PPO 적용](#43-attention-ppo-적용)
   - 4.4 [Self-Play (적대적 강화학습)](#44-self-play-적대적-강화학습)
   - 4.5 [협력 탈출 환경 (Schola)](#45-협력-탈출-환경-schola)
   - 4.6 [COMA 알고리즘과 UE 적용](#46-coma-알고리즘과-ue-적용)
   - 4.7 [MA-POCA 알고리즘](#47-ma-poca-알고리즘)
   - 4.8 [MA-POCA 코드 구현 (UE C++)](#48-ma-poca-코드-구현-ue-c)
   - 4.9 [메이즈 환경 만들기](#49-메이즈-환경-만들기)
   - 4.10 [RND 알고리즘](#410-rnd-알고리즘)
   - 4.11 [투 미션 환경 만들기](#411-투-미션-환경-만들기)
   - 4.12 [Hypernetworks](#412-hypernetworks)
5. [프로젝트](#5-프로젝트)
   - 5.1 [자율주행 연구 환경 구축](#51-자율주행-연구-환경-구축)
   - 5.2 [Unreal ML 유튜브 및 커뮤니티 사례](#52-unreal-ml-유튜브-및-커뮤니티-사례)
   - 5.3 [산업 문제에 UE ML 적용 사례](#53-산업-문제에-ue-ml-적용-사례)
   - 5.4 [상용 게임에 UE ML 적용 사례](#54-상용-게임에-ue-ml-적용-사례)

---

# 1. Unreal Engine ML 개요

## 1.1 Unreal Engine 머신러닝 생태계

Unreal Engine은 Unity와 달리 공식적인 단일 ML-Agents SDK를 제공하지 않습니다. 대신 여러 오픈소스 프로젝트와 에픽게임즈의 실험적 플러그인을 통해 머신러닝/강화학습 환경을 구축할 수 있습니다.

**핵심 구성 요소:**

| 구성 요소 | 설명 |
|-----------|------|
| **UnrealMLAgents** | Unity ML-Agents의 UE 포트, gRPC + Python 트레이너, 가장 성숙한 대안 |
| **Schola (GPUOpen)** | AMD 주도 RL 플러그인, Python Gym 인터페이스, SB3/RLLib 호환 |
| **UE5 Learning Agents** | 에픽게임즈 공식 실험 플러그인, C++ 전용, 소스 빌드 필요 |
| **UE Python API** | 언리얼 에디터 내 Python 실행, 에디터 스크립팅 및 데이터 수집 |
| **External Renderer** | 카메라/렌더 데이터를 외부 파이프로 전송 (ML 입력용) |

## 1.2 지원 알고리즘 (툴킷별)

| 알고리즘 | UnrealMLAgents | Schola | UE5 Learning Agents | 비고 |
|----------|:---:|:---:|:---:|------|
| **PPO** | ✅ (Python SB3) | ✅ (Gym.Env) | ❌ (직접 구현) | On-Policy 기본 |
| **SAC** | ✅ (Python SB3) | ✅ (Gym.Env) | ❌ | Off-Policy, 샘플 효율 |
| **A2C/A3C** | ✅ (SB3) | ✅ | ❌ | 병렬 학습 |
| **DQN** | ✅ (SB3) | ✅ | ❌ | 이산 액션 |
| **MA-POCA** | ❌ (직접 구현) | ❌ | ❌ | Unity 전용, UE에선 COMA/QMIX로 대체 |
| **Self-Play** | ❌ (직접 구현) | ❌ | ❌ | Python 레벨에서 구현 |
| **RND/Curiosity** | ✅ (SB3) | ✅ (커스텀) | ❌ | 내재적 보상 신호 |

## 1.3 학습 방식

Unreal Engine의 ML 학습은 크게 두 가지 방식으로 나뉩니다:

1. **외부 트레이너 (External Trainer)** - UnrealMLAgents/Schola 사용
   - UE 프로세스 ↔ Python 프로세스 간 gRPC/socket 통신
   - Python에서 모든 학습 알고리즘 실행 (PyTorch, SB3)
   - 유연성 높음, 최신 알고리즘 사용 가능

2. **내장 트레이너 (Built-in Trainer)** - UE5 Learning Agents
   - UE5 에디터 내에서 C++ 학습 파이프라인 실행
   - 별도 Python 환경 불필요
   - 단점: 알고리즘 선택 제한적, 소스 빌드 필수

```
[Unreal Engine 5.x]
  ├── UnrealMLAgents Plugin
  │    ├── C++ Agent (UA_AgentBase 상속)
  │    ├── gRPC 통신 (Unity ML-Agents 프로토콜 호환)
  │    └── Behavior Parameters / Decision Requester
  │
  ├── Schola Plugin
  │    ├── C++ Env (AScholaEnv 상속)
  │    ├── Python Gym.Env 래퍼
  │    └── Socket/TCP 통신
  │
  └── UE5 Learning Agents Plugin (Experimental)
       ├── C++ LearningAgentsManager
       ├── C++ Trainer (자체 구현 필요)
       └── Source-built UE5 필수

[Python Environment]
  ├── torch / stable-baselines3 (SB3)
  ├── mlagents (UnrealMLAgents 통신용, Unity ML-Agents 패키지)
  └── schola (Schola 파이썬 클라이언트)
```

---

# 2. 주요 ML 툴킷 비교

## 2.1 UnrealMLAgents

| 항목 | 내용 |
|------|------|
| **저장소** | https://github.com/AlanLaboratory/UnrealMLAgents |
| **최신 버전** | v0.0.9 (2025) |
| **라이선스** | Apache 2.0 |
| **UE 버전** | 5.0 ~ 5.4 (GitHub) / Fab Marketplace 출시 |
| **Python** | Unity ML-Agents 패키지(mlagents) + SB3 |
| **통신 방식** | gRPC (Unity ML-Agents 프로토콜 호환) |
| **에셋/마켓** | Fab (UE 마켓플레이스)에서 배포 |

**특징:**
- Unity ML-Agents의 직접적인 포트 — Unity 경험자에게 친숙
- `UA_AgentBase`, `UA_BehaviorParameters`, `UA_DecisionRequester` 등 C++ 클래스 제공
- Unity ML-Agents의 gRPC 프로토콜을 재사용 → 기존 Python 트레이너 호환
- 별도 Python 패키지 `mlagents`로 학습 실행 (`mlagents-learn`)
- **한계**: 유니티 ML-Agents만큼 성숙하지 않음, 일부 고급 기능 누락 (Ghost Trainer, MA-POCA 등)

### 설치 요약

```bash
# 1. 플러그인 다운로드 (GitHub Releases 또는 Fab)
# 2. 프로젝트 Plugins/ 폴더에 압축 해제
# 3. Python 환경 설정
pip install mlagents==1.1.0 torch stable-baselines3

# 4. 학습 실행 (Unity ML-Agents와 동일한 명령어)
mlagents-learn config/ppo/MyConfig.yaml --run-id=ue_test
```

## 2.2 Schola (GPUOpen)

| 항목 | 내용 |
|------|------|
| **저장소** | https://github.com/GPUOpen-LibrariesAndSDKs/Schola |
| **최신 버전** | v2.0 (2025) |
| **라이선스** | MIT |
| **UE 버전** | 5.5 ~ 5.6 |
| **Python** | gymnasium, stable-baselines3, rllib |
| **통신 방식** | TCP Socket (Python ↔ UE) |
| **특화 분야** | 연구/교육, AMD GPU 최적화 |

**특징:**
- AMD GPUOpen 주도 프로젝트, GPU 가속에 최적화
- Python `gymnasium.Env` 표준 인터페이스 → SB3, RLLib 등 다양한 라이브러리 호환
- `AScholaEnv` 베이스 클래스로 UE 환경 구현
- 멀티 에이전트 지원 (Gym API 확장)
- MIT 라이선스로 상업 사용 자유로움
- **한계**: UE 5.5+ 전용, 커뮤니티가 작음 (약 100 GitHub Stars)

### 설치 요약

```bash
# Python 패키지 설치
pip install schola-gym  # 또는 소스에서 빌드

# Unreal 프로젝트에 Schola 플러그인 추가
# .uproject 파일에 플러그인 등록
```

## 2.3 UE5 Learning Agents (Experimental)

| 항목 | 내용 |
|------|------|
| **저장소** | https://github.com/automathan/ue-la-example |
| **문서** | https://dev.epicgames.com/documentation/ |
| **버전** | Experimental (Unreal Engine 5.2+) |
| **라이선스** | UE EULA |
| **UE 빌드** | 소스 빌드 필수 (GitHub에서 엔진 소스 다운로드) |
| **언어** | C++ 전용 |
| **트레이너** | C++ 내장 (Python 불필요) |

**특징:**
- 에픽게임즈가 공식적으로 UE5에 실험 도입
- `LearningAgentsManager`, `LearningAgentsTrainer`, `LearningAgentsPolicy` 등 C++ 클래스
- C++ 텐서 연산 기반 (Python 의존성 없음)
- 시각적 디버깅: 학습 중 에이전트 행동을 UE 뷰포트에서 실시간 관찰
- **한계**: 소스 빌드 필수, C++ 전용, 실험 단계, 문서 부족, 커뮤니티 극소수

### 설치 요약

```bash
# 1. UE5 소스 빌드 환경 구성 (GitHub epicgames/UnrealEngine)
# 2. 엔진 소스에서 Learning Agents 플러그인 활성화
# 3. 프로젝트에서 플러그인 활성화
# 4. C++ 클래스에서 LearningAgentsManager 사용
```

## 2.4 툴킷 선택 가이드

| 사용자 | 추천 툴킷 | 이유 |
|--------|-----------|------|
| Unity ML-Agents 경험자 | **UnrealMLAgents** | 동일한 gRPC 프로토콜, Python 트레이너, 설정 방식 |
| 연구자 / 교육자 | **Schola** | Gym 표준 인터페이스, 다양한 알고리즘 실험 용이 |
| 순수 UE5 개발자 (C++) | **UE5 Learning Agents** | Python 불필요, UE 에디터 내 모든 작업 |
| AMD GPU 사용자 | **Schola** | AMD GPU 가속 최적화 |
| Fab 에셋 선호 | **UnrealMLAgents** | Fab 마켓플레이스에서 플러그인 구매 가능 |
| 최신 UE 버전 (5.5+) | **Schola** | UE 5.5~5.6 공식 지원 |

---

# 3. 환경 설정 (Environment Setup)

## 3.1 사전 요구사항

### 시스템 요구사항

| 구성 요소 | 요구사항 |
|-----------|----------|
| **OS** | Windows 10/11 (64-bit), Ubuntu 20.04+ |
| **Unreal Engine** | 5.0+ (툴킷별 상이, 5.5 권장) |
| **Python** | Python 3.10 ~ 3.12 |
| **Conda** (선택) | Miniconda 3 (Python 3.10+ 포함) 또는 Anaconda 2024+ |
| **CUDA** (선택) | CUDA 11.8+ (NVIDIA GPU 가속) |
| **메모리** | 최소 16GB RAM (32GB+ 권장) |
| **디스크** | 최소 50GB 여유 공간 (UE 엔진 + 프로젝트) |
| **Visual Studio** | VS 2022 (C++ 워크로드) |

> **Conda vs venv 선택 가이드:**
> - **venv**: Python 기본 내장, 가볍고 단순, UE ML 환경에 충분함
> - **Conda**: 패키지 의존성 충돌 관리에 강력함, CUDA 버전 관리 용이, 재현 가능한 환경(`environment.yml`) 공유에 최적
> - **권장**: 프로젝트가 복잡해지거나 팀 협업이 필요하면 Conda, 단순 개인 학습은 venv로 충분

### 툴킷별 요구사항 매트릭스

| 요구사항 | UnrealMLAgents | Schola | UE5 Learning Agents |
|----------|:---:|:---:|:---:|
| UE 소스 빌드 | ❌ | ❌ | ✅ |
| C++ 컴파일러 | ✅ | ✅ | ✅ |
| Python | ✅ | ✅ | ❌ |
| gRPC | ✅ | ❌ | ❌ |
| CMake | ❌ | ✅ (선택) | ❌ |
| AMD GPU | ❌ | ✅ (선택) | ❌ |
| Conda 환경.yml | ✅ | ✅ | ❌ |

## 3.2 공통 Python 환경 설정

모든 외부 트레이너(UnrealMLAgents, Schola)는 동일한 Python 환경을 사용합니다.

아래 두 가지 방법 중 하나를 선택하세요. 환경 이름은 `ue_ml`로 통일합니다.

---

### 방법 A: venv (Python 내장, 간단 설정)

```bash
# Windows (PowerShell)
python -m venv ue_ml
.\ue_ml\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv ue_ml
source ue_ml/bin/activate
```

---

### 방법 B: Conda (의존성 관리, 재현성 우수)

#### Conda 설치 (아직 없는 경우)

| OS | 다운로드 | 설치 명령어 |
|----|----------|-------------|
| **Windows** | https://docs.conda.io/en/latest/miniconda.html | Miniconda 설치 프로그램 실행 |
| **Linux** | https://docs.conda.io/en/latest/miniconda.html | `bash Miniconda3-latest-Linux-x86_64.sh` |
| **macOS** | https://docs.conda.io/en/latest/miniconda.html | `bash Miniconda3-latest-MacOSX-x86_64.sh` |

설치 후 터미널을 재시작하거나 `conda init`을 실행하세요.

#### Conda 환경 생성

```bash
# 환경 생성 (Python 3.10 지정, UE ML-Agents 호환)
conda create -n ue_ml python=3.10 -y

# 환경 활성화
conda activate ue_ml
```

#### 필수 패키지 설치 (Conda + pip 혼합)

Conda 채널과 pip를 혼용합니다. PyTorch와 CUDA는 conda로, 나머지는 pip로 설치하는 것이 안정적입니다.

```bash
# 1) conda 채널로 설치 (의존성 충돌 방지)
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 cudatoolkit=11.8 -c pytorch -c nvidia -y
conda install numpy pandas matplotlib tensorboard -c conda-forge -y

# 2) pip로 설치 (conda 채널에 없는 패키지)
# UnrealMLAgents용 (Unity ML-Agents 패키지 재사용)
pip install mlagents==1.1.0

# 강화학습
pip install stable-baselines3==2.3.0

# Schola용 (선택)
pip install schola-gym gymnasium

# gRPC (UnrealMLAgents 통신)
pip install protobuf==3.20.3 grpcio==1.53.2

# 유틸리티
pip install onnx onnxruntime pyyaml
```

> **참고:** conda와 pip를 혼용할 때는 **conda 패키지를 먼저** 설치한 후 pip를 실행하세요. 반대 순서로 설치하면 의존성 충돌이 발생할 수 있습니다.

#### 설치 확인

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import stable_baselines3; print(f'SB3 {stable_baselines3.__version__}')"
mlagents-learn --help
```

---

### environment.yml을 사용한 재현 가능한 환경 (Conda)

Conda의 가장 큰 장점은 `environment.yml` 파일 하나로 정확히 동일한 환경을 재현할 수 있다는 점입니다.

#### 1) UnrealMLAgents 전용 환경

`environment_unrealmlagents.yml`:

```yaml
name: ue_ml_unreal
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pytorch=2.1.0
  - torchvision=0.16.0
  - torchaudio=2.1.0
  - cudatoolkit=11.8
  - numpy=1.24.3
  - pandas=2.0.3
  - matplotlib=3.7.2
  - tensorboard=2.14.0
  - pip>=23.0
  - pip:
    - mlagents==1.1.0
    - stable-baselines3==2.3.0
    - protobuf==3.20.3
    - grpcio==1.53.2
    - onnx==1.14.0
    - onnxruntime==1.15.1
    - pyyaml==6.0.1
```

```bash
# 환경 생성
conda env create -f environment_unrealmlagents.yml

# 활성화
conda activate ue_ml_unreal
```

#### 2) Schola 전용 환경

`environment_schola.yml`:

```yaml
name: ue_ml_schola
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pytorch=2.1.0
  - torchvision=0.16.0
  - torchaudio=2.1.0
  - cudatoolkit=11.8
  - numpy=1.24.3
  - pandas=2.0.3
  - matplotlib=3.7.2
  - tensorboard=2.14.0
  - pip>=23.0
  - pip:
    - schola-gym>=2.0.0
    - gymnasium==0.29.0
    - stable-baselines3==2.3.0
    - onnx==1.14.0
    - onnxruntime==1.15.1
```

```bash
conda env create -f environment_schola.yml
conda activate ue_ml_schola
```

#### 3) 올인원 (Full) 환경 — 모든 툴킷 호환

`environment_ue_ml_full.yml`:

```yaml
name: ue_ml_full
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  # --- Python ---
  - python=3.10
  - pip>=23.0

  # --- PyTorch + CUDA ---
  - pytorch=2.1.0
  - torchvision=0.16.0
  - torchaudio=2.1.0
  - cudatoolkit=11.8

  # --- 데이터/시각화 ---
  - numpy=1.24.3
  - pandas=2.0.3
  - matplotlib=3.7.2
  - tensorboard=2.14.0
  - scipy=1.11.1
  - scikit-learn=1.3.0

  # --- C++ 빌드 의존성 (Schola) ---
  - cmake>=3.26
  - ninja>=1.11

  # --- pip 설치 항목 ---
  - pip:
    # UnrealMLAgents (Unity ML-Agents 호환)
    - mlagents==1.1.0

    # 강화학습
    - stable-baselines3==2.3.0
    - sb3-contrib==2.3.0

    # Schola
    - schola-gym>=2.0.0
    - gymnasium==0.29.0

    # gRPC 통신
    - protobuf==3.20.3
    - grpcio==1.53.2

    # 모델 변환/추론
    - onnx==1.14.0
    - onnxruntime==1.15.1
    - onnxruntime-gpu==1.15.1

    # 유틸리티
    - pyyaml==6.0.1
    - tqdm==4.65.0
    - psutil==5.9.5
    - opencv-python==4.8.0.74
```

```bash
conda env create -f environment_ue_ml_full.yml
conda activate ue_ml_full
```

> **설치 시간:** 위 올인원 환경은 모든 패키지를 포함하여 약 5~15분 소요됩니다 (인터넷 속도 및 GPU 사양에 따라 다름).

#### environment.yml 업데이트 및 내보내기

```bash
# 환경에 새 패키지를 추가한 후 yml 업데이트
conda env export -n ue_ml_full > environment_ue_ml_full.yml

# (권장) --from-history: 명시적으로 설치한 패키지만 내보내기 (버전 고정 방지)
conda env export -n ue_ml_full --from-history > environment_ue_ml_full.yml
```

> **팀 협업 팁:** `--from-history`를 사용하면 `pip` 섹션만 수동으로 추가하여, OS/아키텍처에 종속되지 않는 이식성 높은 `environment.yml`을 유지할 수 있습니다.

#### 환경 제거

```bash
conda deactivate
conda env remove -n ue_ml_full
conda info --envs          # 남은 환경 목록 확인
```

---

### Step 3: 설치 확인 (공통)

```bash
# Python 버전
python --version
# → Python 3.10.x

# PyTorch + CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
# → PyTorch 2.1.0, CUDA: True

# Stable-Baselines3
python -c "import stable_baselines3; print(f'SB3 {stable_baselines3.__version__}')"
# → SB3 2.3.0

# UnrealMLAgents CLI
mlagents-learn --help
# → usage: mlagents-learn [-h] ...

# gRPC (UnrealMLAgents 통신 테스트)
python -c "import grpc; print(f'gRPC {grpc.__version__}')"
# → gRPC 1.53.2

# ONNX (모델 변환)
python -c "import onnx; print(f'ONNX {onnx.__version__}')"
# → ONNX 1.14.0
```

### Python 환경 비교 요약

| 항목 | venv | Conda |
|------|------|-------|
| **설치 필요** | Python에 내장 | Miniconda 별도 설치 |
| **환경 위치** | 프로젝트 폴더 내 | `~/miniconda3/envs/` |
| **격리 수준** | pip 패키지만 | Python + 시스템 라이브러리(CUDA 등) |
| **CUDA 관리** | 수동 (pip 인덱스 지정) | conda 채널로 자동 |
| **재현성** | `requirements.txt` (pip freeze) | `environment.yml` (완전 재현) |
| **이식성** | OS 종속적일 수 있음 | 채널 설정으로 OS 독립적 가능 |
| **디스크 사용량** | 작음 (~200MB) | 큼 (~2-5GB, CUDA 포함 시) |
| **권장 사용자** | 간단한 단일 환경 | 팀 협업, 복잡한 의존성 |

## 3.3 UnrealMLAgents 설치 및 설정

### Step 1: 플러그인 설치

**방법 A: Fab 마켓플레이스 (권장)**
1. Epic Games Launcher 실행
2. Fab 마켓플레이스에서 "UnrealMLAgents" 검색
3. 플러그인 구매/다운로드
4. 프로젝트 `Plugins/` 폴더에 설치

**방법 B: GitHub 소스**
```bash
# GitHub에서 플러그인 다운로드
git clone https://github.com/AlanLaboratory/UnrealMLAgents.git
# 또는 Releases 탭에서 최신 버전 다운로드

# 다운로드한 폴더를 프로젝트의 Plugins/UnrealMLAgents/에 복사
```

### Step 2: Unreal 프로젝트 설정

1. Unreal Engine 5.x에서 C++ 프로젝트 생성 (Blank 또는 Third Person 템플릿)
2. 플러그인 활성화 확인: `Edit > Plugins` → "UnrealMLAgents" 검색 후 활성화
3. 에디터 재시작
4. `Build.cs`에 모듈 의존성 추가:

```csharp
// MyProject.Build.cs
PublicDependencyModuleNames.AddRange(new string[] {
    "Core", "CoreUObject", "Engine", "InputCore",
    "UnrealMLAgents"  // UnrealMLAgents 모듈 추가
});
```

### Step 3: Agent 액터 생성

```cpp
// MyRLAgent.h
#pragma once

#include "CoreMinimal.h"
#include "Agents/AgentBase.h"
#include "MyRLAgent.generated.h"

UCLASS()
class MYPROJECT_API AMyRLAgent : public AAgentBase
{
    GENERATED_BODY()

public:
    virtual void CollectObservations(FObservationBuffer& Buffer) override;
    virtual void OnActionReceived(const FActionBuffer& Actions) override;
    virtual void OnEpisodeBegin() override;

private:
    UPROPERTY()
    class UBoxComponent* CollisionBox;

    float EpisodeReward = 0.0f;
};
```

```cpp
// MyRLAgent.cpp
#include "MyRLAgent.h"
#include "Agents/ObservationBuffer.h"
#include "Agents/ActionBuffer.h"

void AMyRLAgent::CollectObservations(FObservationBuffer& Buffer)
{
    // 위치 관측
    Buffer.AddObservation(GetActorLocation().X / 100.0f);
    Buffer.AddObservation(GetActorLocation().Y / 100.0f);
    Buffer.AddObservation(GetActorLocation().Z / 100.0f);

    // 속도 관측
    FVector Velocity = GetVelocity();
    Buffer.AddObservation(Velocity.X);
    Buffer.AddObservation(Velocity.Y);
    Buffer.AddObservation(Velocity.Z);
}

void AMyRLAgent::OnActionReceived(const FActionBuffer& Actions)
{
    // 연속 액션: 이동 (X, Y)
    float MoveX = Actions.GetContinuousAction(0);
    float MoveY = Actions.GetContinuousAction(1);

    FVector NewLocation = GetActorLocation() + FVector(MoveX, MoveY, 0) * 500.0f * GetWorld()->GetDeltaSeconds();
    SetActorLocation(NewLocation);
}

void AMyRLAgent::OnEpisodeBegin()
{
    // 에피소드 초기화: 랜덤 위치에서 시작
    FVector RandomPos = FVector(
        FMath::RandRange(-500.0f, 500.0f),
        FMath::RandRange(-500.0f, 500.0f),
        100.0f
    );
    SetActorLocation(RandomPos);
    EpisodeReward = 0.0f;
}
```

### Step 4: 블루프린트 설정

1. C++ 클래스를 상속받는 블루프린트 생성
2. `UA_BehaviorParameters` 컴포넌트 추가 (AgentBase에 자동 포함)
3. Behavior Parameters 설정:
   - Vector Observation Size: 관측 공간 크기
   - Continuous Actions: 연속 액션 개수
   - Behavior Type: `Default` (학습) / `InferenceOnly` (추론)
4. `UA_DecisionRequester` 컴포넌트 추가 (결정 주기 설정)
5. 씬에 배치

### Step 5: 학습 실행 테스트

```bash
# Unreal 에디터에서 Play 모드 실행 (에이전트가 Default 상태여야 함)
# Python 터미널에서:
mlagents-learn config/ppo/MyConfig.yaml --run-id=ue_first_run
```

**YAML 설정 파일 예시:**

```yaml
behaviors:
  MyRLAgent:               # Behavior Name과 일치해야 함
    trainer_type: ppo
    hyperparameters:
      batch_size: 1024
      buffer_size: 10240
      learning_rate: 3.0e-4
      beta: 5.0e-3
      epsilon: 0.2
      lambd: 0.95
      num_epoch: 3
      learning_rate_schedule: linear
    network_settings:
      normalize: false
      hidden_units: 128
      num_layers: 2
    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0
    max_steps: 5.0e5
    time_horizon: 64
    summary_freq: 10000
```

### Step 6: TensorBoard 모니터링

```bash
tensorboard --logdir=results
```

브라우저에서 `http://localhost:6006` 접속하여 Cumulative Reward 등 확인.

### Step 7: 훈련된 모델 사용

```bash
# 학습 완료 후 .onnx 파일이 results/<run-id>/ 경로에 생성됨
```

1. `.onnx` 파일을 UE 프로젝트 `Content/` 폴더로 임포트
2. Agent의 Behavior Parameters → Model에 할당
3. Behavior Type을 `InferenceOnly`로 변경
4. Play 모드로 동작 확인

## 3.4 Schola 설치 및 설정

### Step 1: 플러그인 빌드

```bash
# Schola 저장소 클론
git clone https://github.com/GPUOpen-LibrariesAndSDKs/Schola.git

# UE 5.5+ 프로젝트의 Plugins/ 폴더에 Schola 플러그인 복사
# 프로젝트 재생성 (Generate Visual Studio Project Files)
```

### Step 2: 환경 클래스 구현

```cpp
// MyScholaEnv.h
#pragma once

#include "CoreMinimal.h"
#include "ScholaEnv.h"
#include "MyScholaEnv.generated.h"

UCLASS()
class MYPROJECT_API AMyScholaEnv : public AScholaEnv
{
    GENERATED_BODY()

public:
    virtual void ConfigureEnv(FEnvInfo& EnvInfo) override;
    virtual FObs GetObservation() override;
    virtual bool IsDone() override;
    virtual float GetReward() override;
    virtual void PerformAction(const FAction& Action) override;
    virtual void ResetEnv() override;
};
```

### Step 3: Python 학습 스크립트

```python
import gymnasium as gym
import schola_gym
from stable_baselines3 import PPO

# Schola 환경 생성
env = gym.make("Schola-MyEnv-v0")

# PPO 학습
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000000)

# 모델 저장
model.save("schola_ppo_model")
```

## 3.5 UE5 Learning Agents 설정

### Step 1: UE5 소스 빌드

```bash
# 1. 에픽게임즈 GitHub 계정 연결
#    https://www.unrealengine.com/en-US/ue-on-github

# 2. UE5 소스 클론
git clone https://github.com/EpicGames/UnrealEngine.git
cd UnrealEngine
git checkout 5.5

# 3. 소스 빌드 (Windows)
Setup.bat
GenerateProjectFiles.bat
# UE5.sln 열고 Development Editor로 빌드
```

### Step 2: 플러그인 활성화

1. 소스 빌드된 UE5 에디터 실행
2. 프로젝트의 `.uproject` 파일에 플러그인 추가:
```json
{
    "Plugins": [
        {
            "Name": "LearningAgents",
            "Enabled": true
        }
    ]
}
```

### Step 3: C++ 학습 매니저 사용

```cpp
// MyLearningManager.h
#include "LearningAgentsManager.h"
#include "LearningAgentsPolicy.h"
#include "LearningAgentsTrainer.h"

UCLASS()
class AMyLearningManager : public ALearningAgentsManager
{
    GENERATED_BODY()

public:
    UPROPERTY()
    ULearningAgentsPolicy* Policy;

    UPROPERTY()
    ULearningAgentsTrainer* Trainer;

    virtual void BeginPlay() override
    {
        Super::BeginPlay();
        
        // 정책과 트레이너 초기화
        Policy = NewObject<ULearningAgentsPolicy>(this);
        Policy->Setup(this);
        
        Trainer = NewObject<ULearningAgentsTrainer>(this);
        Trainer->Setup(this, Policy);
    }

    virtual void Tick(float DeltaTime) override
    {
        Super::Tick(DeltaTime);
        
        // 학습 루프 실행
        if (Trainer)
        {
            Trainer->RunTrainingIteration();
        }
    }
};
```

## 3.6 일반적인 문제 해결

| 문제 | 해결 방법 |
|------|-----------|
| gRPC 통신 오류 | 방화벽 확인, `grpcio==1.53.2` 고정, UE와 Python이 같은 네트워크에 있는지 확인 |
| 플러그인 로드 실패 | UE 버전과 플러그인 버전 일치 확인, C++ 모듈 재빌드 |
| Python/UE 버전 불일치 | 릴리즈 표에서 호환 버전 쌍 확인 |
| CUDA 메모리 부족 | `batch_size` 축소, `num-envs` 감소 |
| ONNX 모델 로드 실패 | 모델이 올바른 Inference Engine 버전으로 export되었는지 확인 |
| UE5 Learning Agents 크래시 | 소스 빌드가 올바른 브랜치인지 확인 (5.2+), 실험 단계임을 인지 |
| **Conda 환경 생성 실패** | `conda clean --all` 실행 후 재시도. `conda update conda -n base`로 conda 자체 업데이트 |
| **Conda + pip 충돌** | conda 패키지를 먼저 설치한 후 pip 실행. 환경을 새로 만들고 순서 준수 |
| **Conda 환경 활성화 안 됨** | `conda init` 실행 후 터미널 재시작. PowerShell 사용 시 `conda activate`를 먼저 실행 |
| **`conda env create` 느림** | `-c conda-forge` 채널 우선 사용, `mamba` 설치 후 `mamba env create`로 대체 (5배 빠름) |
| **PyTorch CUDA 인식 불가** | `conda install cudatoolkit=11.8 -c nvidia` 명시적 설치. `nvidia-smi`로 드라이버 확인 |
| **환경 디스크 부족** | `conda clean -p` (미사용 패키지 삭제), `conda clean -t` (캐시 삭제) |
| **`pip install mlagents` 오류** | Python 3.10 확인, `protobuf==3.20.3` 고정 설치, `pip install --no-cache-dir` 시도 |
| **재현 불가능한 environment.yml** | `conda env export --from-history`로 생성, pip 섹션은 수동 추가 |

---

# 4. 커리큘럼

## 4.1 Dodge 프로젝트 (UnrealMLAgents)

### 4.1.1 개요

**Dodge(피하기) 환경**은 Unity ML-Agents의 DodgeBall 예제를 UnrealMLAgents로 포팅한 프로젝트입니다. 협력과 경쟁 시나리오가 혼합된 다중 에이전트 환경으로, 에이전트가 날아오는 발사체를 피하고 상대를 맞추는 태스크를 학습합니다.

Unity ML-Agents의 DodgeBall과 달리, UnrealMLAgents는 다음 환경을 기본 제공합니다:
- **3DBall (균형 잡기)**: 공을 평평한 판 위에서 균형 유지
- **PushBlock (밀기)**: 블록을 목표 지점으로 이동
- **Hallway (복도)**: 복도 끝까지 이동

Dodge 환경은 위 기본 환경을 확장하여 직접 제작합니다.

### 4.1.2 환경 구성 (C++)

```cpp
// DodgeProjectile.h
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "DodgeProjectile.generated.h"

UCLASS()
class ADodgeProjectile : public AActor
{
    GENERATED_BODY()

public:
    UPROPERTY(EditAnywhere)
    float Speed = 1000.0f;

    UPROPERTY(EditAnywhere)
    float Lifetime = 5.0f;

    virtual void Tick(float DeltaTime) override
    {
        Super::Tick(DeltaTime);
        SetActorLocation(GetActorLocation() + GetActorForwardVector() * Speed * DeltaTime);
        
        Lifetime -= DeltaTime;
        if (Lifetime <= 0.0f)
            Destroy();
    }

    virtual void NotifyActorBeginOverlap(AActor* Other) override
    {
        if (Other->ActorHasTag("Agent"))
        {
            // 에이전트 히트 처리
            // (에이전트 측에서 처리하도록 델리게이트 호출)
            Destroy();
        }
    }
};
```

```cpp
// DodgeAgent.h
#pragma once

#include "CoreMinimal.h"
#include "Agents/AgentBase.h"
#include "DodgeAgent.generated.h"

UCLASS()
class MYPROJECT_API ADodgeAgent : public AAgentBase
{
    GENERATED_BODY()

public:
    UPROPERTY(EditAnywhere)
    float MoveSpeed = 500.0f;

    UPROPERTY(EditAnywhere)
    float RotationSpeed = 180.0f;

    int32 Health = 1;

    virtual void CollectObservations(FObservationBuffer& Buffer) override;
    virtual void OnActionReceived(const FActionBuffer& Actions) override;
    virtual void OnEpisodeBegin() override;

    UFUNCTION()
    void OnHitByProjectile();
};
```

```cpp
// DodgeAgent.cpp
#include "DodgeAgent.h"
#include "Agents/ObservationBuffer.h"
#include "Agents/ActionBuffer.h"
#include "Kismet/GameplayStatics.h"

void ADodgeAgent::CollectObservations(FObservationBuffer& Buffer)
{
    // 자신의 위치와 속도
    Buffer.AddObservation(GetActorLocation().X / 1000.0f);
    Buffer.AddObservation(GetActorLocation().Y / 1000.0f);
    FVector Velocity = GetVelocity();
    Buffer.AddObservation(Velocity.X);
    Buffer.AddObservation(Velocity.Y);

    // 주변 발사체 감지 (Ray Perception 대체)
    TArray<AActor*> Projectiles;
    UGameplayStatics::GetAllActorsOfClass(GetWorld(), ADodgeProjectile::StaticClass(), Projectiles);

    for (auto* Proj : Projectiles)
    {
        FVector RelPos = Proj->GetActorLocation() - GetActorLocation();
        Buffer.AddObservation(RelPos.X / 2000.0f);
        Buffer.AddObservation(RelPos.Y / 2000.0f);
        Buffer.AddObservation(FVector::DotProduct(Proj->GetVelocity(), GetActorForwardVector()) / 1000.0f);
    }
}

void ADodgeAgent::OnActionReceived(const FActionBuffer& Actions)
{
    float MoveX = Actions.GetContinuousAction(0);
    float MoveY = Actions.GetContinuousAction(1);
    float Rotate = Actions.GetContinuousAction(2);

    FVector DeltaMove = FVector(MoveX, MoveY, 0.0f) * MoveSpeed * GetWorld()->GetDeltaSeconds();
    SetActorLocation(GetActorLocation() + DeltaMove);

    FRotator NewRot = GetActorRotation();
    NewRot.Yaw += Rotate * RotationSpeed * GetWorld()->GetDeltaSeconds();
    SetActorRotation(NewRot);

    // 시간 패널티 (효율적 행동 유도)
    AddReward(-0.001f);
}

void ADodgeAgent::OnEpisodeBegin()
{
    FVector RandomPos = FVector(
        FMath::RandRange(-800.0f, 800.0f),
        FMath::RandRange(-800.0f, 800.0f),
        100.0f
    );
    SetActorLocation(RandomPos);
    Health = 1;
}

void ADodgeAgent::OnHitByProjectile()
{
    Health--;
    if (Health <= 0)
    {
        AddReward(-1.0f);  // 사망 패널티
        EndEpisode();
    }
    else
    {
        AddReward(-0.1f);  // 피격 패널티
    }
}
```

### 4.1.3 YAML 설정 (UnrealMLAgents)

```yaml
behaviors:
  DodgeAgent:
    trainer_type: ppo
    hyperparameters:
      batch_size: 128
      buffer_size: 2048
      learning_rate: 3.0e-4
      beta: 5.0e-3
      epsilon: 0.2
      lambd: 0.95
      num_epoch: 3
    network_settings:
      normalize: true
      hidden_units: 256
      num_layers: 2
    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0
    max_steps: 2.0e6
    time_horizon: 128
    summary_freq: 10000
```

### 4.1.4 학습 실행

```bash
# Unreal 에디터에서 레벨 로드 후 Play
mlagents-learn config/ppo/DodgeAgent.yaml --run-id=dodge_ue_01 --num-envs=4
```

### 4.1.5 Schola로 Dodge 환경 구현 (대안)

Schola를 사용하면 Gym 표준 인터페이스로 Dodge 환경을 구현할 수 있습니다:

```python
# schola_dodge_env.py
import gymnasium as gym
import numpy as np
from schola_gym import ScholaEnv

class DodgeEnv(ScholaEnv):
    def __init__(self):
        super().__init__(
            env_name="DodgeEnv",
            observation_space=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(12,)
            ),
            action_space=gym.spaces.Box(
                low=-1.0, high=1.0, shape=(3,)  # MoveX, MoveY, Rotate
            )
        )

    def step(self, action):
        # UE 환경에 액션 전송
        self.send_action(action)
        
        # 다음 상태, 보상, 종료, 정보 수신
        obs, reward, terminated, truncated, info = self.receive_observation()
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None):
        return self.receive_reset()
```

**학습 스크립트:**

```python
from stable_baselines3 import PPO
from schola_dodge_env import DodgeEnv

env = DodgeEnv()
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tb_logs/")
model.learn(total_timesteps=500000)
model.save("dodge_ppo_schola")
```

### 4.1.6 환경 확장

- **팀 모드 추가**: 두 팀으로 나누어 팀전 구현
- **발사체 패턴 변경**: 다양한 발사 속도/궤적
- **장애물 추가**: 맵에 장애물 배치하여 전략적 회피 유도
- **파워업 아이템**: 속도 증가, 실드 등 아이템 추가

---

## 4.2 PPO 알고리즘

### 4.2.1 개요

**PPO (Proximal Policy Optimization)** 는 OpenAI에서 개발한 강화학습 알고리즘으로, Unreal Engine ML 환경에서 가장 널리 사용되는 기본 트레이너입니다. Unity ML-Agents와 달리, Unreal에서는 PPO가 UE 내장 알고리즘이 아니라 Python (SB3)에서 실행됩니다.

- **논문**: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
- **UE 적용**: Python SB3 PPO + UnrealMLAgents/Schola 통신
- **유형**: On-Policy, Actor-Critic

### 4.2.2 Unreal Engine에서의 PPO 학습 구조

```
┌──────────────────────────────────────────────────┐
│              Unreal Engine 5.x                    │
│                                                   │
│  ┌──────────────┐     ┌─────────────────────┐     │
│  │  Agent Actor  │────►│  Observation Buffer │     │
│  │  (C++/BP)     │◄────│  Action Buffer      │     │
│  └──────┬───────┘     └─────────┬───────────┘     │
│         │                       │                  │
│         ▼                       ▼                  │
│  ┌──────────────────────────────────────┐          │
│  │       gRPC / TCP Socket 통신          │          │
│  └──────────────────────────────────────┘          │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│              Python Environment                   │
│                                                   │
│  ┌──────────────────────────────────────┐          │
│  │        Stable-Baselines3 PPO         │          │
│  │  ┌────────┐  ┌──────────┐           │          │
│  │  │ Actor  │  │  Critic  │           │          │
│  │  │Network │  │  Network │           │          │
│  │  └────────┘  └──────────┘           │          │
│  └──────────────────────────────────────┘          │
│                                                   │
│  ┌──────────────────────────────────────┐          │
│  │           TensorBoard Logger          │          │
│  └──────────────────────────────────────┘          │
└──────────────────────────────────────────────────┘
```

### 4.2.3 핵심 개념

#### 목적 함수 (Clipped Surrogate Objective)

```
L^CLIP(θ) = Ê_t[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]
```

- `r_t(θ)`: 확률비 (새 정책 / 이전 정책)
- `Â_t`: 이점 함수 (Advantage Function)
- `ε`: 클리핑 범위 (보통 0.2)

#### PPO vs 다른 알고리즘

| 알고리즘 | 샘플 효율 | 학습 안정성 | UE 적용 난이도 |
|----------|-----------|------------|----------------|
| **PPO (SB3)** | 중간 | 높음 | 낮음 (SB3 기본 지원) |
| **SAC (SB3)** | 높음 | 중간 | 낮음 (SB3 기본 지원) |
| **DQN (SB3)** | 낮음 | 낮음 | 낮음 (이산 액션 전용) |
| **A2C (SB3)** | 중간 | 중간 | 낮음 (SB3 기본 지원) |

### 4.2.4 SB3 PPO 하이퍼파라미터

| 파라미터 | 설명 | 일반 범위 | 기본값 |
|----------|------|-----------|--------|
| `batch_size` | 미니배치 크기 | 64 ~ 4096 | 64 |
| `n_steps` | 스텝 수 (buffer_size 역할) | 128 ~ 2048 | 2048 |
| `learning_rate` | 학습률 | 1e-5 ~ 1e-3 | 3e-4 |
| `ent_coef` | 엔트로피 계수 (beta) | 1e-4 ~ 0.01 | 0.0 |
| `clip_range` | 클리핑 범위 (epsilon) | 0.1 ~ 0.3 | 0.2 |
| `gae_lambda` | GAE 파라미터 (lambd) | 0.9 ~ 0.99 | 0.95 |
| `n_epochs` | 에포크 수 (num_epoch) | 3 ~ 20 | 10 |
| `gamma` | 할인 계수 | 0.8 ~ 0.9997 | 0.99 |
| `net_arch` | 네트워크 구조 | [64,64] ~ [512,512] | dict(pi=[64,64], vf=[64,64]) |

### 4.2.5 Python PPO 학습 스크립트 (UnrealMLAgents)

```python
# train_ppo_ue.py
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import torch

def make_env():
    """UnrealMLAgents 환경 생성"""
    unity_env = UnityEnvironment(
        file_name=None,  # None = 에디터 연결
        seed=42,
        side_channels=[]
    )
    unity_env.reset()
    
    # Behavior 이름 자동 탐지
    behavior_name = list(unity_env.behavior_specs.keys())[0]
    
    # Unity Gym 래퍼 사용
    gym_env = UnityToGymWrapper(
        unity_env,
        worker_id=0,
        flatten_branched=True,
        allow_multiple_obs=False
    )
    return gym_env

# 환경 생성
env = DummyVecEnv([make_env])

# PPO 모델 생성
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.001,
    verbose=1,
    tensorboard_log="./tb_logs/"
)

# 학습
model.learn(total_timesteps=1000000)

# 모델 저장
model.save("ue_ppo_model")

# ONNX 내보내기 (Unreal에서 추론용)
import mlagents.trainers.torch_utils as torch_utils
dummy_input = torch.randn(1, env.observation_space.shape[0])
torch.onnx.export(
    model.policy.actor_net,
    dummy_input,
    "ue_ppo_model.onnx",
    input_names=["obs"],
    output_names=["action"],
    dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}}
)
```

### 4.2.6 Schola PPO 학습 스크립트 (Gym 인터페이스)

```python
# train_ppo_schola.py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import schola_gym

def make_env():
    return schola_gym.make("MyCustomEnv-v0")

env = DummyVecEnv([make_env])

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./tb_logs/schola_ppo/"
)
model.learn(total_timesteps=1000000)
model.save("schola_ppo_model")
```

### 4.2.7 PPO 학습 모범 사례 (UE)

1. **관측 정규화**: `normalize: true` 또는 SB3 `VecNormalize` 사용
2. **네트워크 구조**: 간단한 환경 → `[64,64]`, 복잡한 환경 → `[256,256]`
3. **n_steps 조정**: 너무 작으면 샘플 다양성 부족, 너무 크면 학습 속도 저하
4. **멀티 환경**: `--num-envs=4` 또는 SB3 `SubprocVecEnv`로 학습 속도 향상
5. **Timestep 일관성**: UE의 `DeltaSeconds`를 고려한 액션 스케일링

### 4.2.8 PPO 학습 커맨드 모음

```bash
# UnrealMLAgents 기본 학습
mlagents-learn config/ppo/MyConfig.yaml --run-id=ue_ppo_01

# 멀티 환경 학습
mlagents-learn config/ppo/MyConfig.yaml --run-id=ue_ppo_02 --num-envs=4

# 기존 모델 불러와서 추가 학습
mlagents-learn config/ppo/MyConfig.yaml --run-id=ue_ppo_02 --resume

# SB3 직접 학습 (Python)
python train_ppo_ue.py

# TensorBoard 모니터링
tensorboard --logdir=./tb_logs/
```

---

## 4.3 Attention PPO 적용

### 4.3.1 Attention Mechanism 개요

어텐션 메커니즘은 입력 데이터의 중요한 부분에 "주목"할 수 있게 하는 신경망 기술입니다. Unreal Engine 환경에서는 에이전트가 주변 객체(발사체, 장애물, 팀원 등) 중 중요한 것에 선택적으로 집중할 수 있게 합니다.

### 4.3.2 Self-Attention의 핵심 개념

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

- **Q (Query)**: 현재 요소가 "질문"하는 벡터
- **K (Key)**: 다른 요소들이 "답변"하는 키 벡터
- **V (Value)**: 실제 정보가 담긴 값 벡터
- **d_k**: 스케일링 팩터

### 4.3.3 SB3 PPO 네트워크에 Attention 레이어 추가

UnrealMLAgents/Schola는 Python에서 네트워크를 구성하므로, Unity ML-Agents보다 유연하게 어텐션을 적용할 수 있습니다.

```python
# attention_ppo.py
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

class AttentionNetwork(nn.Module):
    """Self-Attention 특징 추출기"""
    def __init__(self, feature_dim, hidden_dim=128, num_heads=4):
        super().__init__()
        self.embedding = nn.Linear(feature_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, feature_dim)
        x = self.embedding(x)
        attn_out, _ = self.attention(x, x, x)
        x = self.layer_norm(x + attn_out)  # Residual connection
        x = x.mean(dim=1)  # Sequence pooling
        return self.output(x)


class AttentionActorCriticPolicy(ActorCriticPolicy):
    """어텐션을 사용하는 Actor-Critic 정책"""
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        # 네트워크 구성을 위해 features_dim 설정
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=dict(pi=[128], vf=[128]),
            **kwargs
        )

    def _build_mlp_extractor(self):
        """MLP 추출기를 Attention 추출기로 대체"""
        class AttentionExtractor(nn.Module):
            def __init__(self, feature_dim):
                super().__init__()
                self.shared_net = AttentionNetwork(
                    feature_dim=feature_dim,
                    hidden_dim=128,
                    num_heads=4
                )
                self.pi_net = nn.Identity()
                self.vf_net = nn.Identity()
                
            def forward(self, features):
                return self.shared_net(features), self.shared_net(features)

        return AttentionExtractor(self.features_dim)


# 커스텀 정책으로 PPO 생성
model = PPO(
    policy=AttentionActorCriticPolicy,
    env=env,
    verbose=1,
    tensorboard_log="./tb_logs/attention_ppo/"
)
model.learn(total_timesteps=500000)
```

### 4.3.4 UE C++에서 BufferSensor 역할 (가변 길이 관측)

UnrealMLAgents는 Unity의 `BufferSensor`와 동등한 기능이 없으므로, 직접 구현해야 합니다. Schola를 사용하면 Python Gym에서 유연하게 처리 가능합니다.

```cpp
// UE C++에서 가변 길이 관측 처리
void AMyAgent::CollectObservations(FObservationBuffer& Buffer)
{
    // 고정 관측: 자신의 상태
    Buffer.AddObservation(GetActorLocation().X);
    Buffer.AddObservation(GetActorLocation().Y);
    Buffer.AddObservation(GetVelocity().Size());

    // 가변 길이 관측: 주변 액터 (최대 10개로 패딩)
    TArray<AActor*> NearbyActors;
    GetOverlappingActors(NearbyActors, AActor::StaticClass());

    int32 Count = 0;
    for (auto* Actor : NearbyActors)
    {
        if (Count >= 10) break;
        FVector RelPos = Actor->GetActorLocation() - GetActorLocation();
        Buffer.AddObservation(RelPos.X / 1000.0f);
        Buffer.AddObservation(RelPos.Y / 1000.0f);
        Count++;
    }

    // 패딩: 나머지 0으로 채움
    for (int32 i = Count; i < 10; i++)
    {
        Buffer.AddObservation(0.0f);
        Buffer.AddObservation(0.0f);
    }

    // 활성 객체 수 추가 전달
    Buffer.AddObservation(static_cast<float>(Count) / 10.0f);
}
```

### 4.3.5 Attention PPO 적용 단계

1. **환경 분석**: 에이전트가 관측해야 할 중요 객체 식별
2. **관측 공간 설계**: 최대 객체 수 결정, 패딩 전략 수립
3. **Attention 네트워크 구현**: SB3 커스텀 정책으로 어텐션 레이어 추가
4. **학습**: 수정된 네트워크로 PPO 학습 실행
5. **평가**: 기본 PPO 대비 성능 비교 (수렴 속도, 최종 보상)

> **참고**: Unreal 환경에서 Attention PPO는 객체 수가 가변적이거나 부분 관측성이 강한 환경에서 특히 효과적입니다. SB3의 유연한 네트워크 구성 덕분에 Unity ML-Agents보다 적용이 쉽습니다.

---

## 4.4 Self-Play (적대적 강화학습)

### 4.4.1 Self-Play 개요

Self-Play는 에이전트가 자기 자신의 과거 정책들을 상대로 학습하는 방식입니다. Unreal Engine에서는 Unity ML-Agents의 Ghost Trainer와 달리, Python 레벨에서 직접 Self-Play를 구현해야 합니다.

### 4.4.2 Python Self-Play 구현 (SB3)

UnrealMLAgents/Schola는 자체 Ghost Trainer가 없으므로, Python에서 SB3로 Self-Play 루프를 직접 구현합니다:

```python
# self_play_ue.py
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class SelfPlayManager:
    """Unreal Engine Self-Play 관리자"""

    def __init__(self, env, save_dir="./self_play_checkpoints/"):
        self.env = env
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.current_model = None
        self.opponent_pool = []
        self.max_pool_size = 10
        self.window = 10        # 저장할 과거 정책 수
        self.swap_steps = 10000  # 상대방 변경 간격
        self.step_counter = 0

    def initialize(self):
        """첫 번째 모델 초기화"""
        self.current_model = PPO(
            "MlpPolicy", self.env, verbose=0
        )

    def select_opponent(self):
        """과거 정책 중 상대방 선택"""
        if not self.opponent_pool:
            return self.current_model

        # 50% 확률로 최신 모델, 50% 확률로 과거 모델
        if np.random.random() < 0.5 and len(self.opponent_pool) > 1:
            idx = np.random.randint(0, len(self.opponent_pool) - 1)
            return self.opponent_pool[idx]
        else:
            return self.current_model

    def save_checkpoint(self, steps):
        """현재 모델을 과거 정책 풀에 저장"""
        path = os.path.join(self.save_dir, f"checkpoint_{steps}.zip")
        self.current_model.save(path)

        # 저장된 모델을 풀에 추가
        checkpoint_model = PPO.load(path, env=self.env)
        self.opponent_pool.append(checkpoint_model)

        # 풀 크기 제한
        if len(self.opponent_pool) > self.max_pool_size:
            old_model = self.opponent_pool.pop(0)
            del old_model

    def train_iteration(self, timesteps=10000):
        """Self-Play 학습 반복"""
        self.step_counter += timesteps

        # 상대방 선택 (에이전트 정책으로 설정)
        opponent = self.select_opponent()

        # 현재 모델 학습 (opponent와 대결)
        self.current_model.learn(total_timesteps=timesteps)

        # 주기적으로 체크포인트 저장
        if self.step_counter % self.swap_steps < timesteps:
            self.save_checkpoint(self.step_counter)

        return self.current_model


# 실행 예시
env = DummyVecEnv([lambda: make_unreal_env()])
sp_manager = SelfPlayManager(env)
sp_manager.initialize()

for iteration in range(100):
    model = sp_manager.train_iteration(timesteps=10000)
    print(f"Iteration {iteration}: pool size = {len(sp_manager.opponent_pool)}")

model.save("ue_self_play_final")
```

### 4.4.3 UnrealMLAgents에서 Self-Play 환경 구성

Unity ML-Agents의 Ghost Trainer가 YAML에 `self_play:` 섹션을 추가하는 것과 달리, UnrealMLAgents에서는 Python 스크립트로 직접 제어해야 합니다.

```yaml
# UnrealMLAgents YAML (Self-Play 없음 - Python에서 구현)
behaviors:
  CompetitiveAgent:
    trainer_type: ppo
    hyperparameters:
      batch_size: 128
      buffer_size: 2048
      learning_rate: 3.0e-4
      beta: 1.0e-2
      epsilon: 0.2
      lambd: 0.95
      num_epoch: 3
    network_settings:
      normalize: true
      hidden_units: 256
      num_layers: 2
    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0
    max_steps: 1.0e7
    time_horizon: 64
    summary_freq: 10000
```

### 4.4.4 Schola Self-Play (Gym 인터페이스)

Schola의 Gym 인터페이스는 Self-Play 구현이 더 간단합니다:

```python
import gymnasium as gym
import schola_gym
from stable_baselines3 import PPO
import numpy as np

# 경쟁 환경: Agent 1과 Agent 2가 동일한 환경에서 대결
env = schola_gym.make("CompetitiveEnv-v0")

# 메인 에이전트
main_agent = PPO("MlpPolicy", env, verbose=0)
# 상대방 (과거 스냅샷)
opponent = PPO("MlpPolicy", env, verbose=0)

# Self-Play 루프
for episode in range(1000):
    obs, _ = env.reset()
    done = False
    
    while not done:
        # 메인 에이전트 액션
        action_main, _ = main_agent.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, _ = env.step(action_main)
        done = terminated or truncated
    
    # 에피소드 종료 후: 승패에 따라 보상 로깅
    if reward > 0:
        print(f"Episode {episode}: Win")
    else:
        print(f"Episode {episode}: Lose")
    
    # 주기적으로 상대방 업데이트
    if episode % 50 == 0:
        opponent = PPO.load(main_agent.save("temp_model"), env=env)
```

### 4.4.5 Self-Play 모범 사례 (UE)

1. **Python 제어**: Unity ML-Agents와 달리 UE는 Python에서 Self-Play를 완전히 제어 (YAML 설정 불가)
2. **상대 풀 크기**: 10~20개 체크포인트 유지. 너무 많으면 메모리 부족
3. **최신 모델 대결 비율**: 50% 확률로 과거 정책, 50% 확률로 최신 정책과 대결
4. **ELO/Elo Rating**: Python에서 직접 ELO 점수를 계산하여 성능 추적
5. **멀티 프로세스**: SB3 `SubprocVecEnv`로 여러 게임 인스턴스를 병렬 실행

---

## 4.5 협력 탈출 환경 (Schola)

### 4.5.1 개요

협력 탈출(Cooperative Escape) 환경은 여러 에이전트가 협력하여 탈출구를 찾는 다중 에이전트 태스크입니다. Unity ML-Agents의 Dungeon Escape에 대응되는 UE 환경입니다.

UnrealMLAgents는 Unity의 `SimpleMultiAgentGroup`을 지원하지 않으므로, **Schola**의 Gym 인터페이스를 사용하는 것을 권장합니다.

### 4.5.2 Schola 협력 환경 구현

```cpp
// UE C++: CooperativeEscapeEnv.h
#pragma once

#include "CoreMinimal.h"
#include "ScholaEnv.h"
#include "CooperativeEscapeEnv.generated.h"

UCLASS()
class ACooperativeEscapeEnv : public AScholaEnv
{
    GENERATED_BODY()

public:
    UPROPERTY(EditAnywhere)
    int32 NumAgents = 2;

    UPROPERTY(EditAnywhere)
    float EscapeReward = 10.0f;

    UPROPERTY(EditAnywhere)
    float StepPenalty = -0.01f;

    virtual void ConfigureEnv(FEnvInfo& EnvInfo) override
    {
        EnvInfo.NumAgents = NumAgents;
        EnvInfo.ObservationSize = 12;  // 위치(3) + 속도(3) + 목표방향(3) + 팀원방향(3)
        EnvInfo.ActionSize = 2;         // MoveX, MoveY
        EnvInfo.UseDiscreteActions = false;
    }

    virtual FObs GetObservation() override
    {
        FObs Obs;
        Obs.Data.SetNum(NumAgents * 12);

        for (int32 i = 0; i < NumAgents; i++)
        {
            AActor* Agent = Agents[i];
            FVector Loc = Agent->GetActorLocation();
            FVector Vel = Agent->GetVelocity();

            int32 Offset = i * 12;
            Obs.Data[Offset + 0] = Loc.X / 1000.0f;
            Obs.Data[Offset + 1] = Loc.Y / 1000.0f;
            Obs.Data[Offset + 2] = Loc.Z / 1000.0f;
            Obs.Data[Offset + 3] = Vel.X / 500.0f;
            Obs.Data[Offset + 4] = Vel.Y / 500.0f;
            Obs.Data[Offset + 5] = Vel.Z / 500.0f;
            // ... 추가 관측
        }
        return Obs;
    }

    virtual float GetReward() override
    {
        return StepPenalty;  // 기본 단계 패널티
    }

    virtual bool IsDone() override
    {
        // 모든 에이전트가 탈출구에 도달했는지 확인
        for (auto* Agent : Agents)
        {
            if (FVector::Dist(Agent->GetActorLocation(), EscapeExitLocation) < 200.0f)
            {
                AddGlobalReward(EscapeReward);  // 공동 보상
                return true;
            }
        }
        return false;
    }
};
```

### 4.5.3 Python 협력 학습 (SB3 + Schola)

Schola의 Gym 인터페이스를 사용하여 협력 학습을 실행합니다:

```python
# cooperative_escape.py
import gymnasium as gym
from stable_baselines3 import PPO
from sb3_contrib import QRDQN  # 또는 다른 알고리즘

# Schola 협력 환경
env = gym.make("Schola-CoopEnv-v0")

# PPO로 학습 (다중 에이전트를 하나의 정책으로)
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./tb_logs/coop_escape/"
)
model.learn(total_timesteps=2000000)
model.save("coop_escape_ppo")
```

### 4.5.4 UnrealMLAgents로 협력 학습 (수동 구현)

UnrealMLAgents는 `SimpleMultiAgentGroup`이 없으므로, 협력 보상을 수동으로 구현해야 합니다:

```cpp
void ACoopAgent::OnActionReceived(const FActionBuffer& Actions)
{
    // 액션 처리
    ProcessMovement(Actions);

    // 협력 보상: 모든 에이전트가 목표에 가까워질수록 보상
    float TeamDistance = 0.0f;
    TArray<AActor*> AllAgents;
    UGameplayStatics::GetAllActorsOfClass(GetWorld(), ACoopAgent::StaticClass(), AllAgents);

    for (auto* OtherAgent : AllAgents)
    {
        float DistToGoal = FVector::Dist(OtherAgent->GetActorLocation(), GoalLocation);
        TeamDistance += DistToGoal;
    }

    // 평균 거리가 줄어들면 보상 (모든 에이전트에 동일 보상)
    float AvgDistance = TeamDistance / AllAgents.Num();
    float Reward = (PreviousAvgDistance - AvgDistance) * 0.01f;
    PreviousAvgDistance = AvgDistance;

    // 모든 에이전트에 동일한 보상 추가
    for (auto* Agent : AllAgents)
    {
        Cast<ACoopAgent>(Agent)->AddReward(Reward);
    }

    AddReward(-0.001f);  // 단계 패널티
}
```

### 4.5.5 협력 환경 팁

1. **그룹 보상 공유**: 모든 에이전트가 같은 보상을 받도록 설계 (Unity MA-POCA와 동일 개념)
2. **개인 보상 혼합**: 협력만으로는 학습이 어려울 경우 개인 보상 추가
3. **커리큘럼**: 2인 → 4인 → 8인 순으로 에이전트 수 증가
4. **통신 채널**: 에이전트 간 정보 공유를 위한 관측 공간 설계

---

## 4.6 COMA 알고리즘과 UE 적용

### 4.6.1 개요

**COMA (Counterfactual Multi-Agent Policy Gradients)** 는 다중 에이전트 강화학습을 위한 Actor-Critic 알고리즘입니다. Unity ML-Agents는 COMA를 내장하고 있지 않으며, UE에서도 마찬가지로 Python에서 직접 구현해야 합니다.

- **논문**: "Counterfactual Multi-Agent Policy Gradients" (Foerster et al., 2018)
- **적용 분야**: 협력 다중 에이전트 태스크
- **핵심**: Counterfactual Baseline을 통한 크레딧 할당

### 4.6.2 Python COMA 구현 (SB3 기반)

```python
# coma_ue.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stable_baselines3.common.vec_env import VecEnv

class CentralizedCritic(nn.Module):
    """중앙 비평자: 모든 에이전트의 관측과 행동을 입력받음"""
    def __init__(self, num_agents, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        input_dim = num_agents * (obs_dim + action_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Q(s, a)
        )

    def forward(self, observations, actions):
        # observations: (batch, num_agents, obs_dim)
        # actions: (batch, num_agents, action_dim)
        batch_size = observations.shape[0]
        x = torch.cat([observations, actions], dim=-1)
        x = x.view(batch_size, -1)  # Flatten all agents
        return self.net(x)


class COMATrainer:
    """COMA 트레이너: UE 환경과 연동"""

    def __init__(self, env: VecEnv, num_agents, obs_dim, action_dim, lr=3e-4):
        self.env = env
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # 각 에이전트의 Actor
        self.actors = [PPO("MlpPolicy", env, verbose=0) for _ in range(num_agents)]

        # 중앙 비평자
        self.critic = CentralizedCritic(num_agents, obs_dim, action_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def compute_counterfactual_advantage(self, observations, actions, reward):
        """
        Counterfactual Advantage 계산
        A_i(s, a) = Q(s, a) - Σ_{a'_i} π_i(a'_i | τ_i) Q(s, (a_{-i}, a'_i))
        """
        obs_tensor = torch.FloatTensor(observations)
        act_tensor = torch.FloatTensor(actions)

        # Q(s, a): 전체 행동에 대한 Q-값
        q_value = self.critic(obs_tensor, act_tensor)

        advantages = []
        for i in range(self.num_agents):
            # 에이전트 i의 행동을 고정하고 다른 행동들에 대해 주변화
            # 실제로는 가능한 모든 행동에 대해 샘플링 또는 적분 필요
            # 여기서는 간략화를 위해 Monte Carlo 근사 사용
            advantage = reward - q_value.mean().item()  # 단순화된 버전
            advantages.append(advantage)

        return advantages

    def train_step(self, batch):
        """학습 스텝"""
        obs, actions, rewards = batch

        # Counterfactual Advantage 계산
        advantages = self.compute_counterfactual_advantage(obs, actions, rewards)

        # 각 Actor 업데이트 (자신의 Advantage로)
        for i in range(self.num_agents):
            # 각 에이전트의 정책 업데이트
            pass  # 실제 구현 필요

        # Critic 업데이트
        pred_q = self.critic(torch.FloatTensor(obs), torch.FloatTensor(actions))
        target_q = torch.FloatTensor(rewards).unsqueeze(-1)
        critic_loss = F.mse_loss(pred_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
```

### 4.6.3 COMA vs MA-POCA (UE 관점)

| 특징 | COMA | MA-POCA (Unity) | UE 적용 |
|------|------|-----------------|---------|
| 비평자 구조 | 중앙 Q-함수 | 그룹 단위 Q-함수 | 직접 구현 필요 |
| 크레딧 할당 | Counterfactual Baseline | 그룹 보상 공유 | Python 레벨 구현 |
| 확장성 | 에이전트 증가 시 차원 폭발 | 그룹 단위로 확장 가능 | Schola로 일부 해결 |
| UE 구현 난이도 | 높음 (Python 전체 구현) | UE 미지원 | Schola Gym으로 완화 |

> **권장**: UE에서는 COMA 대신 Python + SB3 멀티에이전트 라이브러리(Tianshou, PettingZoo 등)를 사용하는 것이 실용적입니다.

---

## 4.7 MA-POCA 알고리즘

### 4.7.1 개요

**MA-POCA (Multi-Agent POCA)** 는 Unity ML-Agents가 제공하는 다중 에이전트 협력 학습 알고리즘으로, UE에서는 공식 지원되지 않습니다. 이 섹션에서는 UE에서 MA-POCA와 유사한 기능을 구현하는 방법을 다룹니다.

### 4.7.2 UE에서 MA-POCA 개념 구현

```python
# ue_mapoca_like.py
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

class MAPOCALikeTrainer:
    """
    MA-POCA 유사 트레이너 (Unreal Engine용)
    - 그룹 단위 보상 공유
    - 중앙 비평자 대신 그룹 보상을 각 에이전트에 전파
    """

    def __init__(self, env: VecEnv, num_agents, policy_cls=PPO):
        self.env = env
        self.num_agents = num_agents

        # 모든 에이전트가 동일한 정책 공유 (파라미터 공유)
        self.shared_policy = policy_cls("MlpPolicy", env, verbose=1)

    def train(self, total_timesteps):
        """그룹 보상 기반 학습"""
        self.shared_policy.learn(total_timesteps=total_timesteps)

    def predict(self, observation, deterministic=True):
        """모든 에이전트에 동일 정책 적용"""
        return self.shared_policy.predict(observation, deterministic=deterministic)
```

### 4.7.3 UE C++ 그룹 보상 시스템

UnrealMLAgents/Schola는 Unity의 `SimpleMultiAgentGroup`이 없으므로 직접 구현해야 합니다:

```cpp
// UE C++ 그룹 보상 시스템
UCLASS()
class UGroupRewardManager : public UObject
{
    GENERATED_BODY()

public:
    TArray<AAgentBase*> AgentGroup;
    float AccumulatedGroupReward = 0.0f;

    void RegisterAgent(AAgentBase* Agent)
    {
        AgentGroup.Add(Agent);
    }

    void UnregisterAgent(AAgentBase* Agent)
    {
        AgentGroup.Remove(Agent);
    }

    void AddGroupReward(float Reward)
    {
        AccumulatedGroupReward += Reward;
    }

    void DistributeGroupReward()
    {
        float PerAgentReward = AccumulatedGroupReward / FMath::Max(1, AgentGroup.Num());
        for (auto* Agent : AgentGroup)
        {
            Agent->AddReward(PerAgentReward);
        }
        AccumulatedGroupReward = 0.0f;
    }

    void EndGroupEpisode()
    {
        DistributeGroupReward();
        for (auto* Agent : AgentGroup)
        {
            Agent->EndEpisode();
        }
    }
};
```

### 4.7.4 YAML 설정

```yaml
# MA-POCA-like를 위한 YAML (UnrealMLAgents용)
# 실제 MA-POCA가 아니며, PPO로 그룹 보상을 학습합니다.
behaviors:
  CooperativeAgent:
    trainer_type: ppo
    hyperparameters:
      batch_size: 128
      buffer_size: 2048
      learning_rate: 3.0e-4
      beta: 5.0e-3
      epsilon: 0.2
      lambd: 0.95
      num_epoch: 3
    network_settings:
      normalize: true
      hidden_units: 256
      num_layers: 2
    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0
    max_steps: 5.0e6
    time_horizon: 128
    summary_freq: 10000
```

### 4.7.5 적용 환경

- **협력 푸시 블록**: 여러 에이전트가 블록을 목표 지점으로 이동
- **협력 탈출**: 팀원과 협력하여 탈출구 탐색
- **팀 전투**: 팀 단위 전투에서 협력 전술 학습

---

## 4.8 MA-POCA 코드 구현 (UE C++)

### 4.8.1 C++ 공동 작업 에이전트 구현

```cpp
// CooperativeAgent.h
#pragma once

#include "CoreMinimal.h"
#include "Agents/AgentBase.h"
#include "CooperativeAgent.generated.h"

UCLASS()
class MYPROJECT_API ACooperativeAgent : public AAgentBase
{
    GENERATED_BODY()

public:
    UPROPERTY(EditAnywhere)
    class ATargetPoint* TargetLocation;

    virtual void CollectObservations(FObservationBuffer& Buffer) override;
    virtual void OnActionReceived(const FActionBuffer& Actions) override;
    virtual void OnEpisodeBegin() override;

    void SetGroupManager(class UGroupRewardManager* Manager) { GroupManager = Manager; }

private:
    UPROPERTY()
    class UGroupRewardManager* GroupManager;

    UPROPERTY()
    class URigidBodyComponent* PhysicsBody;

    bool bReachedTarget = false;
};
```

```cpp
// CooperativeAgent.cpp
#include "CooperativeAgent.h"
#include "GroupRewardManager.h"
#include "Agents/ObservationBuffer.h"
#include "Agents/ActionBuffer.h"
#include "Engine/TargetPoint.h"

void ACooperativeAgent::CollectObservations(FObservationBuffer& Buffer)
{
    // 자신의 위치와 속도
    FVector Loc = GetActorLocation();
    Buffer.AddObservation(Loc.X / 1000.0f);
    Buffer.AddObservation(Loc.Y / 1000.0f);
    Buffer.AddObservation(Loc.Z / 1000.0f);

    FVector Vel = GetVelocity();
    Buffer.AddObservation(Vel.X / 500.0f);
    Buffer.AddObservation(Vel.Y / 500.0f);
    Buffer.AddObservation(Vel.Z / 500.0f);

    // 목표 위치까지의 상대 벡터
    if (TargetLocation)
    {
        FVector TargetDir = TargetLocation->GetActorLocation() - Loc;
        Buffer.AddObservation(TargetDir.X / 2000.0f);
        Buffer.AddObservation(TargetDir.Y / 2000.0f);
        Buffer.AddObservation(TargetDir.Z / 2000.0f);
    }
}

void ACooperativeAgent::OnActionReceived(const FActionBuffer& Actions)
{
    float MoveX = Actions.GetContinuousAction(0);
    float MoveY = Actions.GetContinuousAction(1);

    FVector Delta = FVector(MoveX, MoveY, 0.0f) * 500.0f * GetWorld()->GetDeltaSeconds();
    SetActorLocation(GetActorLocation() + Delta, true);

    // 단계별 패널티
    AddReward(-0.001f);

    // 그룹 보상 분배 (주기적)
    if (GroupManager && GetWorld()->GetTimeSeconds() - LastDistributionTime > 0.5f)
    {
        GroupManager->DistributeGroupReward();
        LastDistributionTime = GetWorld()->GetTimeSeconds();
    }

    // 목표 도달 체크
    if (TargetLocation && FVector::Dist(GetActorLocation(), TargetLocation->GetActorLocation()) < 100.0f && !bReachedTarget)
    {
        bReachedTarget = true;
        if (GroupManager)
        {
            GroupManager->AddGroupReward(5.0f);  // 큰 그룹 보상
            GroupManager->EndGroupEpisode();
        }
    }
}

void ACooperativeAgent::OnEpisodeBegin()
{
    FVector RandomPos = FVector(
        FMath::RandRange(-800.0f, 800.0f),
        FMath::RandRange(-800.0f, 800.0f),
        100.0f
    );
    SetActorLocation(RandomPos);
    bReachedTarget = false;
}
```

### 4.8.2 Python (SB3) 학습 스크립트

```python
# train_coop_ue.py
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

def create_ue_env(worker_id=0):
    """UnrealMLAgents 환경 생성"""
    unity_env = UnityEnvironment(
        file_name=None,
        worker_id=worker_id,
        seed=42 + worker_id,
        side_channels=[]
    )
    behavior_name = list(unity_env.behavior_specs.keys())[0]
    
    return UnityToGymWrapper(
        unity_env,
        worker_id=worker_id,
        flatten_branched=True,
        allow_multiple_obs=False
    )

# 멀티 환경 학습 (4개 병렬)
if __name__ == "__main__":
    env = SubprocVecEnv([lambda i=i: create_ue_env(i) for i in range(4)])

    # 콜백 설정
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./checkpoints/",
        name_prefix="ue_coop_model"
    )

    eval_callback = EvalCallback(
        env,
        best_model_save_path="./best_model/",
        log_path="./eval_logs/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    # PPO 학습
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        verbose=1,
        tensorboard_log="./tb_logs/coop_ue/"
    )

    model.learn(
        total_timesteps=3000000,
        callback=[checkpoint_callback, eval_callback]
    )

    model.save("ue_coop_final")
```

### 4.8.3 학습 결과 확인

TensorBoard에서 확인할 주요 메트릭:

| 메트릭 | 설명 |
|--------|------|
| `rollout/ep_rew_mean` | 에피소드 평균 보상 (증가 추세 확인) |
| `rollout/ep_len_mean` | 에피소드 평균 길이 (감소 추세 확인) |
| `train/policy_gradient_loss` | 정책 그래디언트 손실 |
| `train/value_loss` | 가치 함수 손실 |
| `train/entropy_loss` | 정책 엔트로피 (탐험 정도) |

---

## 4.9 메이즈 환경 만들기

### 4.9.1 개요

메이즈(미로) 환경은 강화학습 에이전트가 미로를 탐색하여 목표 지점에 도달하는 태스크입니다. Unreal Engine에서는 절차적 메시 생성(Procedural Mesh)을 사용하여 동적으로 미로를 생성할 수 있습니다.

### 4.9.2 UE 절차적 메이즈 생성 (C++)

```cpp
// MazeGenerator.h
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "MazeGenerator.generated.h"

USTRUCT()
struct FMazeCell
{
    GENERATED_BODY()
    bool bVisited = false;
    bool bWallTop = true;
    bool bWallBottom = true;
    bool bWallLeft = true;
    bool bWallRight = true;
};

UCLASS()
class MYPROJECT_API AMazeGenerator : public AActor
{
    GENERATED_BODY()

public:
    UPROPERTY(EditAnywhere, Category = "Maze")
    int32 MazeWidth = 10;

    UPROPERTY(EditAnywhere, Category = "Maze")
    int32 MazeHeight = 10;

    UPROPERTY(EditAnywhere, Category = "Maze")
    float CellSize = 200.0f;

    UPROPERTY(EditAnywhere, Category = "Maze")
    UMaterialInterface* WallMaterial;

    UPROPERTY(EditAnywhere, Category = "Maze")
    UMaterialInterface* FloorMaterial;

    UFUNCTION(BlueprintCallable, Category = "Maze")
    void GenerateMaze();

    UFUNCTION(BlueprintCallable, Category = "Maze")
    FVector GetStartLocation() const { return StartLocation; }

    UFUNCTION(BlueprintCallable, Category = "Maze")
    FVector GetGoalLocation() const { return GoalLocation; }

private:
    TArray<TArray<FMazeCell>> MazeGrid;
    void CarvePath(int32 X, int32 Y);
    void BuildMesh();
    FVector CellToWorld(int32 X, int32 Y) const;

    FVector StartLocation;
    FVector GoalLocation;
    class UProceduralMeshComponent* MeshComponent;
};
```

```cpp
// MazeGenerator.cpp
#include "MazeGenerator.h"
#include "ProceduralMeshComponent.h"

void AMazeGenerator::GenerateMaze()
{
    // 그리드 초기화
    MazeGrid.SetNum(MazeWidth);
    for (int32 x = 0; x < MazeWidth; x++)
    {
        MazeGrid[x].SetNum(MazeHeight);
        for (int32 y = 0; y < MazeHeight; y++)
        {
            MazeGrid[x][y] = FMazeCell();
        }
    }

    // DFS로 미로 생성 (시작: 0,0)
    CarvePath(0, 0);

    // 메시 빌드
    BuildMesh();

    // 시작/목표 위치 설정
    StartLocation = CellToWorld(0, 0) + FVector(0, 0, 100);
    GoalLocation = CellToWorld(MazeWidth - 1, MazeHeight - 1) + FVector(0, 0, 100);
}

void AMazeGenerator::CarvePath(int32 X, int32 Y)
{
    MazeGrid[X][Y].bVisited = true;

    // 방향 섞기 (랜덤)
    TArray<TPair<int32, int32>> Directions = {
        {0, 1}, {0, -1}, {1, 0}, {-1, 0}
    };
    for (int32 i = Directions.Num() - 1; i > 0; i--)
    {
        int32 j = FMath::RandRange(0, i);
        Directions.Swap(i, j);
    }

    for (auto& Dir : Directions)
    {
        int32 NewX = X + Dir.Key;
        int32 NewY = Y + Dir.Value;

        if (NewX >= 0 && NewX < MazeWidth && NewY >= 0 && NewY < MazeHeight && !MazeGrid[NewX][NewY].bVisited)
        {
            // 벽 제거
            if (Dir.Key == 1)  // 오른쪽
            {
                MazeGrid[X][Y].bWallRight = false;
                MazeGrid[NewX][NewY].bWallLeft = false;
            }
            else if (Dir.Key == -1)  // 왼쪽
            {
                MazeGrid[X][Y].bWallLeft = false;
                MazeGrid[NewX][NewY].bWallRight = false;
            }
            else if (Dir.Value == 1)  // 위
            {
                MazeGrid[X][Y].bWallTop = false;
                MazeGrid[NewX][NewY].bWallBottom = false;
            }
            else if (Dir.Value == -1)  // 아래
            {
                MazeGrid[X][Y].bWallBottom = false;
                MazeGrid[NewX][NewY].bWallTop = false;
            }

            CarvePath(NewX, NewY);
        }
    }
}

void AMazeGenerator::BuildMesh()
{
    MeshComponent = NewObject<UProceduralMeshComponent>(this);
    MeshComponent->RegisterComponent();

    TArray<FVector> Vertices;
    TArray<int32> Triangles;
    TArray<FVector2D> UVs;
    TArray<FVector> Normals;

    float HalfCell = CellSize / 2.0f;

    for (int32 x = 0; x < MazeWidth; x++)
    {
        for (int32 y = 0; y < MazeHeight; y++)
        {
            FVector Center = CellToWorld(x, y);

            // 바닥
            Vertices.Add(Center + FVector(-HalfCell, -HalfCell, 0));
            Vertices.Add(Center + FVector( HalfCell, -HalfCell, 0));
            Vertices.Add(Center + FVector( HalfCell,  HalfCell, 0));
            Vertices.Add(Center + FVector(-HalfCell,  HalfCell, 0));

            int32 BaseIdx = Vertices.Num() - 4;
            Triangles.Add(BaseIdx + 0); Triangles.Add(BaseIdx + 2); Triangles.Add(BaseIdx + 1);
            Triangles.Add(BaseIdx + 0); Triangles.Add(BaseIdx + 3); Triangles.Add(BaseIdx + 2);

            // 벽 (필요한 경우에만 생성)
            float WallHeight = 200.0f;
            float WallThick = 20.0f;

            if (MazeGrid[x][y].bWallTop)
            {
                // 상단 벽 생성 (Vertex 추가)
                // ... (실제 구현에서는 충분한 Vertex/삼각형 추가)
            }
            // ... 나머지 방향 벽 처리
        }
    }

    MeshComponent->CreateMeshSection(0, Vertices, Triangles, Normals, UVs, TArray<FColor>(), TArray<FProcMeshTangent>(), true);
    MeshComponent->SetMaterial(0, FloorMaterial);
}

FVector AMazeGenerator::CellToWorld(int32 X, int32 Y) const
{
    return GetActorLocation() + FVector(X * CellSize, Y * CellSize, 0);
}
```

### 4.9.3 Maze Agent 구현

```cpp
// MazeAgent.h
#pragma once

#include "CoreMinimal.h"
#include "Agents/AgentBase.h"
#include "MazeAgent.generated.h"

UCLASS()
class MYPROJECT_API AMazeAgent : public AAgentBase
{
    GENERATED_BODY()

public:
    virtual void CollectObservations(FObservationBuffer& Buffer) override;
    virtual void OnActionReceived(const FActionBuffer& Actions) override;
    virtual void OnEpisodeBegin() override;

    UFUNCTION()
    void OnReachedGoal();

    UFUNCTION()
    void OnHitWall();

private:
    UPROPERTY()
    class AMazeGenerator* MazeGenerator;

    FVector StartPosition;
    TArray<float> PreviousDistances;
};
```

```cpp
// MazeAgent.cpp
#include "MazeAgent.h"
#include "MazeGenerator.h"
#include "Agents/ObservationBuffer.h"
#include "Agents/ActionBuffer.h"

void AMazeAgent::OnEpisodeBegin()
{
    if (!MazeGenerator)
    {
        MazeGenerator = Cast<AMazeGenerator>(
            UGameplayStatics::GetActorOfClass(GetWorld(), AMazeGenerator::StaticClass())
        );
    }

    // 새 에피소드마다 새로운 미로 생성
    MazeGenerator->GenerateMaze();

    // 에이전트 위치 초기화
    SetActorLocation(MazeGenerator->GetStartLocation());
    SetActorRotation(FRotator::ZeroRotator);
}

void AMazeAgent::CollectObservations(FObservationBuffer& Buffer)
{
    // Ray Perception: 8개 방향으로 레이캐스트
    TArray<FVector> Directions = {
        FVector(1,0,0), FVector(-1,0,0), FVector(0,1,0), FVector(0,-1,0),
        FVector(0.707,0.707,0), FVector(0.707,-0.707,0),
        FVector(-0.707,0.707,0), FVector(-0.707,-0.707,0)
    };

    FVector Start = GetActorLocation();
    FCollisionQueryParams Params;
    Params.AddIgnoredActor(this);

    for (auto& Dir : Directions)
    {
        FHitResult Hit;
        FVector End = Start + Dir * 1000.0f;

        if (GetWorld()->LineTraceSingleByChannel(Hit, Start, End, ECC_WorldStatic, Params))
        {
            Buffer.AddObservation(Hit.Distance / 1000.0f);  // 정규화
        }
        else
        {
            Buffer.AddObservation(1.0f);  // 최대 거리
        }
    }

    // 방향 벡터 (바라보는 방향)
    Buffer.AddObservation(GetActorForwardVector().X);
    Buffer.AddObservation(GetActorForwardVector().Y);
}

void AMazeAgent::OnActionReceived(const FActionBuffer& Actions)
{
    float MoveX = Actions.GetContinuousAction(0);
    float MoveY = Actions.GetContinuousAction(1);

    FVector Delta = FVector(MoveX, MoveY, 0.0f) * 400.0f * GetWorld()->GetDeltaSeconds();
    SetActorLocation(GetActorLocation() + Delta, true);

    // 목표 방향 회전
    FVector ToGoal = MazeGenerator->GetGoalLocation() - GetActorLocation();
    FRotator TargetRot = ToGoal.Rotation();
    SetActorRotation(FMath::RInterpTo(GetActorRotation(), TargetRot, GetWorld()->GetDeltaSeconds(), 2.0f));

    // 단계 패널티
    AddReward(-0.001f);

    // 탐험 보상: 새 위치 방문 시 보상
    float DistToGoal = FVector::Dist(GetActorLocation(), MazeGenerator->GetGoalLocation());
    if (!PreviousDistances.Contains(DistToGoal))
    {
        PreviousDistances.Add(DistToGoal);
        AddReward(0.01f);  // 탐험 보상
    }
}

void AMazeAgent::OnReachedGoal()
{
    AddReward(10.0f);  // 목표 달성 큰 보상
    EndEpisode();
}

void AMazeAgent::OnHitWall()
{
    AddReward(-0.05f);  // 벽 충돌 패널티
}
```

### 4.9.4 YAML 설정

```yaml
behaviors:
  MazeAgent:
    trainer_type: ppo
    hyperparameters:
      batch_size: 256
      buffer_size: 4096
      learning_rate: 3.0e-4
      beta: 5.0e-3
      epsilon: 0.2
      lambd: 0.95
      num_epoch: 3
    network_settings:
      normalize: true
      hidden_units: 256
      num_layers: 3
      memory:               # LSTM 사용 (미로 탐색에 효과적)
        sequence_length: 64
        memory_size: 256
    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0
    max_steps: 5.0e6
    time_horizon: 256
    summary_freq: 10000
```

### 4.9.5 메이즈 학습 팁 (UE)

1. **절차적 생성**: 매 에피소드마다 새로운 미로 생성 → 일반화 능력 향상
2. **커리큘럼 학습**: 5x5 → 10x10 → 15x15 순으로 난이도 증가
3. **탐험 보상**: 방문하지 않은 셀 탐험 시 추가 보상 (희소 보상 문제 완화)
4. **LSTM 사용**: 미로 탐색은 메모리가 중요 → `memory` 설정 활성화
5. **Ray Perception**: UE의 `LineTraceByChannel`을 사용한 Ray Perception 구현
6. **Procedural Mesh**: UE의 `ProceduralMeshComponent`로 동적 메시 생성

---

## 4.10 RND 알고리즘

### 4.10.1 개요

**RND (Random Network Distillation)** 는 희소 보상 환경에서 탐험을 장려하는 방법입니다. UE 환경에서는 SB3의 내장 RND(또는 유사 기능)을 사용하거나 직접 구현할 수 있습니다.

- **논문**: "Exploration by Random Network Distillation" (Burda et al., 2019)
- **SB3 지원**: SB3는 RND를 기본 제공하지 않지만, 커스텀 콜백으로 구현 가능

### 4.10.2 Python RND 구현 (SB3 + UE 연동)

```python
# rnd_ue.py
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class RNDNetwork(nn.Module):
    """RND: Target Network + Predictor Network"""

    def __init__(self, obs_dim, hidden_dim=256, encoding_dim=128):
        super().__init__()

        # Target Network: 랜덤 초기화 후 고정
        self.target = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, encoding_dim)
        )

        # Predictor Network: 학습 가능
        self.predictor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, encoding_dim)
        )

        # Target 가중치 고정
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, obs):
        with torch.no_grad():
            target_feat = self.target(obs)
        pred_feat = self.predictor(obs)

        # 예측 오차 = 내재적 보상
        intrinsic_reward = torch.mean((target_feat - pred_feat) ** 2, dim=-1, keepdim=True)
        return intrinsic_reward, pred_feat


class RNDCallback(BaseCallback):
    """RND 내재적 보상을 PPO 학습에 통합하는 콜백"""

    def __init__(self, obs_dim, rnd_lr=1e-3, intrinsic_coeff=0.1, verbose=0):
        super().__init__(verbose)
        self.rnd = RNDNetwork(obs_dim)
        self.rnd_optimizer = torch.optim.Adam(self.rnd.predictor.parameters(), lr=rnd_lr)
        self.intrinsic_coeff = intrinsic_coeff

    def _on_step(self) -> bool:
        # 로컬 버퍼에서 관측 가져오기
        if hasattr(self.model, 'rollout_buffer') and len(self.model.rollout_buffer.observations) > 0:
            obs_tensor = torch.FloatTensor(self.model.rollout_buffer.observations[-1])

            # 내재적 보상 계산
            intrinsic_reward, pred_feat = self.rnd(obs_tensor)

            # Predictor Network 업데이트 (Target 모방 학습)
            with torch.no_grad():
                target_feat = self.rnd.target(obs_tensor)
            predictor_loss = nn.MSELoss()(pred_feat, target_feat)

            self.rnd_optimizer.zero_grad()
            predictor_loss.backward()
            self.rnd_optimizer.step()

            # 내재적 보상을 extrinsic 보상에 추가 (실제로는 rollout buffer 수정 필요)
            intrinsic_reward_value = intrinsic_reward.item() * self.intrinsic_coeff

            if self.verbose > 0:
                self.logger.record("rnd/intrinsic_reward", intrinsic_reward_value)
                self.logger.record("rnd/predictor_loss", predictor_loss.item())

        return True


# 사용 예시
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

unity_env = UnityEnvironment(file_name=None)
behavior_name = list(unity_env.behavior_specs.keys())[0]
env = UnityToGymWrapper(unity_env, flatten_branched=True)

obs_dim = env.observation_space.shape[0]

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./tb_logs/rnd_ue/"
)

# RND 콜백 추가
rnd_callback = RNDCallback(obs_dim=obs_dim, intrinsic_coeff=0.1)
model.learn(total_timesteps=1000000, callback=rnd_callback)
```

### 4.10.3 RND 하이퍼파라미터

| 파라미터 | 설명 | 권장값 |
|----------|------|--------|
| `intrinsic_coeff` | 내재적 보상 강도 (η) | 0.01 ~ 1.0 |
| `encoding_dim` | 특징 벡터 크기 | 128 ~ 512 |
| `rnd_lr` | RND Predictor 학습률 | 1.0e-4 ~ 1.0e-3 |
| `hidden_dim` | RND 네트워크 은닉층 크기 | 128 ~ 256 |

### 4.10.4 RND 사용 타이밍 (UE 환경)

RND가 특히 효과적인 UE 환경:
- **희소 보상 환경**: 보상이 매우 드물게 주어지는 환경 (미로 찾기, 보물 찾기)
- **복잡한 3D 환경**: UE의 대규모 3D 환경에서 탐험 가이드
- **장기 과제**: 에피소드가 긴 태스크 (자율주행, 건설)
- **절차적 생성 맵**: 매 에피소드마다 새로운 맵이 생성되는 환경

### 4.10.5 RND vs Curiosity (ICM) in UE

| 특징 | RND | ICM (Intrinsic Curiosity Module) |
|------|-----|----------------------------------|
| 예측 대상 | 랜덤 타겟 특징 | 다음 상태 (forward dynamics) |
| UE 구현 난이도 | 중간 | 중간-높음 |
| 계산 비용 | 낮음 | 중간 |
| Noisy-TV 문제 | 강건함 | 취약함 |
| SB3 호환 | 콜백으로 구현 | 직접 구현 필요 |

---

## 4.11 투 미션 환경 만들기

### 4.11.1 개요

"투 미션(Two Missions)" 환경은 에이전트가 두 가지 상이한 임무를 수행해야 하는 환경입니다. Unreal Engine에서는 **Gameplay Ability System (GAS)** 또는 단순 상태 머신(State Machine)을 사용하여 미션 전환을 구현할 수 있습니다.

### 4.11.2 UE 미션 시스템 설계

```cpp
// MissionComponent.h
#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "MissionComponent.generated.h"

UENUM(BlueprintType)
enum class EMissionType : uint8
{
    Collect UMETA(DisplayName = "Collection"),
    Puzzle  UMETA(DisplayName = "Puzzle"),
    Both    UMETA(DisplayName = "Both")
};

UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class MYPROJECT_API UMissionComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UPROPERTY(EditAnywhere, BlueprintReadOnly)
    EMissionType CurrentMission = EMissionType::Collect;

    UPROPERTY(EditAnywhere)
    int32 CoinsToCollect = 5;

    UPROPERTY(EditAnywhere)
    float MissionCompleteBonus = 2.0f;

    // 미션 A: 수집
    int32 CoinsCollected = 0;
    bool bMissionAComplete = false;

    // 미션 B: 퍼즐
    bool bPuzzleSolved = false;
    bool bMissionBComplete = false;

    UFUNCTION(BlueprintCallable)
    void OnCoinCollected()
    {
        if (CurrentMission == EMissionType::Collect)
        {
            CoinsCollected++;
            if (CoinsCollected >= CoinsToCollect)
            {
                bMissionAComplete = true;
                CurrentMission = EMissionType::Puzzle;
                // 보너스 보상은 에이전트가 처리
                OnMissionChanged.Broadcast(EMissionType::Puzzle);
            }
        }
    }

    UFUNCTION(BlueprintCallable)
    void OnPuzzleSolved()
    {
        if (CurrentMission == EMissionType::Puzzle)
        {
            bPuzzleSolved = true;
            bMissionBComplete = true;
            OnAllMissionsComplete.Broadcast();
        }
    }

    bool IsAllComplete() const
    {
        return bMissionAComplete && bPuzzleSolved;
    }

    DECLARE_MULTICAST_DELEGATE_OneParam(FOnMissionChanged, EMissionType);
    FOnMissionChanged OnMissionChanged;

    DECLARE_MULTICAST_DELEGATE(FOnAllMissionsComplete);
    FOnAllMissionsComplete OnAllMissionsComplete;
};
```

```cpp
// DualMissionAgent.cpp (핵심 부분)
#include "DualMissionAgent.h"
#include "MissionComponent.h"

void ADualMissionAgent::CollectObservations(FObservationBuffer& Buffer)
{
    // 공통 관측
    Buffer.AddObservation(GetActorLocation().X / 1000.0f);
    Buffer.AddObservation(GetActorLocation().Y / 1000.0f);

    // 현재 미션 정보 (원-핫 인코딩)
    Buffer.AddObservation(MissionComp->CurrentMission == EMissionType::Collect ? 1.0f : 0.0f);
    Buffer.AddObservation(MissionComp->CurrentMission == EMissionType::Puzzle ? 1.0f : 0.0f);

    // 미션 진행 상태
    Buffer.AddObservation(MissionComp->bMissionAComplete ? 1.0f : 0.0f);
    Buffer.AddObservation(MissionComp->bMissionBComplete ? 1.0f : 0.0f);

    // 미션 A: 남은 코인 수
    Buffer.AddObservation(static_cast<float>(MissionComp->CoinsToCollect - MissionComp->CoinsCollected) / 10.0f);

    // 미션 B: 퍼즐 해결 상태
    Buffer.AddObservation(MissionComp->bPuzzleSolved ? 1.0f : 0.0f);

    // 목표 방향
    FVector TargetDir = GetTargetDirection();
    Buffer.AddObservation(TargetDir.X);
    Buffer.AddObservation(TargetDir.Y);
}

void ADualMissionAgent::OnActionReceived(const FActionBuffer& Actions)
{
    // 이동 처리
    float MoveX = Actions.GetContinuousAction(0);
    float MoveY = Actions.GetContinuousAction(1);
    SetActorLocation(GetActorLocation() + FVector(MoveX, MoveY, 0.0f) * 500.0f * GetWorld()->GetDeltaSeconds());

    // 단계 패널티
    AddReward(-0.001f);

    // 미션 완료 체크
    if (MissionComp->IsAllComplete() && !bAllCompleteRewarded)
    {
        bAllCompleteRewarded = true;
        AddReward(5.0f);
        EndEpisode();
    }
}
```

### 4.11.3 시나리오 설계

#### 시나리오 A: 순차적 미션

```
[에피소드 시작]
    │
    ▼
┌────────────────┐
│ 미션 A: 코인 수집 │─── 5개 코인 수집 완료 ───►┌────────────────┐
│  (탐색/수집)     │                           │ 미션 B: 퍼즐 해결 │─── 완료 ───► 보상 + 종료
└────────────────┘                            └────────────────┘
```

#### 시나리오 B: 동시 미션 (멀티 에이전트)

```
┌────────────────────┐
│ 에이전트 1: 코인 수집 │
└────────┬───────────┘
         │ 협력 보상 공유
┌────────▼───────────┐
│  에이전트 2: 퍼즐 해결 │
└────────────────────┘
```

### 4.11.4 YAML 설정 (멀티태스크)

```yaml
behaviors:
  DualMissionAgent:
    trainer_type: ppo
    hyperparameters:
      batch_size: 256
      buffer_size: 4096
      learning_rate: 3.0e-4
      beta: 5.0e-3
      epsilon: 0.2
      lambd: 0.95
      num_epoch: 3
      learning_rate_schedule: linear
    network_settings:
      normalize: true
      hidden_units: 256
      num_layers: 3
    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0
      curiosity:           # 호기심 기반 탐험 보조
        gamma: 0.99
        strength: 0.1
        encoding_size: 128
        learning_rate: 1.0e-3
    max_steps: 1.0e7
    time_horizon: 128
    summary_freq: 10000
```

### 4.11.5 확장 아이디어

1. **랜덤 미션 순서**: 매 에피소드마다 미션 순서 랜덤화
2. **블루프린트 전환**: C++가 아닌 블루프린트로 미션 로직 구성 (비개발자 친화적)
3. **멀티 에이전트 협력**: 각 에이전트가 다른 미션 담당
4. **레벨 시퀀스**: UE `Level Streaming`으로 각 미션 공간 분리
5. **계층적 학습**: 상위 정책이 미션 선택, 하위 정책이 실행

---

## 4.12 Hypernetworks

### 4.12.1 개요

**Hypernetwork**는 다른 신경망의 가중치를 생성하는 신경망입니다. UE 환경에서는 Python에서 Hypernetwork를 구현하여 UE 에이전트의 정책을 생성할 수 있습니다.

### 4.12.2 Python Hypernetwork 구현 (SB3 연동)

```python
# hypernetwork_ue.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

class HyperNetwork(nn.Module):
    """주 네트워크의 가중치를 생성하는 Hypernetwork"""

    def __init__(self, embedding_dim=32, obs_dim=12, hidden_dim=128, action_dim=4):
        super().__init__()

        # Hypernetwork - 소규모
        self.hyper = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )

        # Actor 가중치 생성기
        self.actor_weight = nn.Linear(128, hidden_dim * obs_dim)
        self.actor_bias = nn.Linear(128, hidden_dim)
        self.actor_out_weight = nn.Linear(128, action_dim * hidden_dim)
        self.actor_out_bias = nn.Linear(128, action_dim)

        # Critic 가중치 생성기
        self.critic_weight = nn.Linear(128, hidden_dim * obs_dim)
        self.critic_bias = nn.Linear(128, hidden_dim)
        self.critic_out_weight = nn.Linear(128, 1 * hidden_dim)
        self.critic_out_bias = nn.Linear(128, 1)

    def generate_actor(self, z, obs):
        """Hypernetwork가 생성한 가중치로 Actor 네트워크 구성"""
        h = self.hyper(z)

        w1 = self.actor_weight(h).view(-1, obs.shape[-1])
        b1 = self.actor_bias(h)
        w2 = self.actor_out_weight(h).view(-1, 128)
        b2 = self.actor_out_bias(h)

        x = F.linear(obs, w1, b1)
        x = F.relu(x)
        action = F.linear(x, w2, b2)
        return action

    def generate_critic(self, z, obs):
        """Hypernetwork가 생성한 가중치로 Critic 네트워크 구성"""
        h = self.hyper(z)

        w1 = self.critic_weight(h).view(-1, obs.shape[-1])
        b1 = self.critic_bias(h)
        w2 = self.critic_out_weight(h).view(-1, 128)
        b2 = self.critic_out_bias(h)

        x = F.linear(obs, w1, b1)
        x = F.relu(x)
        value = F.linear(x, w2, b2)
        return value


class HyperActorCriticPolicy(ActorCriticPolicy):
    """Hypernetwork 기반 Actor-Critic 정책"""

    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        self.hypernet = HyperNetwork(
            embedding_dim=32,
            obs_dim=observation_space.shape[0],
            action_dim=action_space.shape[0]
        )
        # 에이전트 임베딩 (멀티 에이전트 파라미터 생성)
        self.agent_embedding = nn.Embedding(10, 32)

    def forward(self, obs, agent_id=0, deterministic=False):
        z = self.agent_embedding(torch.tensor([agent_id], device=obs.device))
        actions = self.hypernet.generate_actor(z, obs)
        values = self.hypernet.generate_critic(z, obs)
        return actions, values
```

### 4.12.3 Hypernetwork 활용 시나리오 (UE)

| 시나리오 | 설명 | 장점 |
|----------|------|------|
| **멀티 에이전트** | 하나의 Hypernetwork가 여러 에이전트 정책 생성 | 파라미터 효율적, 에이전트 간 지식 공유 |
| **환경 조건 적응** | 환경 파라미터를 Hypernetwork 입력으로 사용 | 동적 난이도 조절 |
| **전이 학습** | 학습된 Hypernetwork를 새 태스크에 적용 | 빠른 적응 |
| **메타 학습** | 다양한 태스크에서 학습 후 새로운 태스크에 빠르게 적응 | 범용성 |

### 4.12.4 Hypernetwork 변형: HyperMARL in UE

HyperMARL (2024)의 개념을 UE에 적용:

```python
class HyperMARL:
    """Hypernetwork 기반 멀티 에이전트 RL"""

    def __init__(self, num_agents, obs_dim, action_dim, env):
        self.num_agents = num_agents
        self.env = env
        self.hyper_policy = HyperActorCriticPolicy(...)

    def train_step(self, observations):
        """각 에이전트의 파라미터를 Hypernetwork로 생성"""
        total_loss = 0

        for agent_id in range(self.num_agents):
            # 에이전트별 파라미터 생성
            agent_obs = observations[agent_id]
            actions, values = self.hyper_policy(agent_obs, agent_id=agent_id)

            # 각 에이전트의 정책 손실 계산
            loss = self.compute_loss(actions, values, agent_id)
            total_loss += loss

        # Hypernetwork만 업데이트 (공유)
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
```

---

# 5. 프로젝트

## 5.1 자율주행 연구 환경 구축

### 5.1.1 개요

Unreal Engine은 고품질 렌더링과 물리 엔진을 바탕으로 자율주행 시뮬레이션에 널리 사용됩니다. Unity ML-Agents와 달리, UE는 **CARLA**, **AirSim** 등 자율주행 특화 플러그인이 별도로 존재하며, UE 내장 ML 툴킷으로도 자율주행 환경을 구축할 수 있습니다.

### 5.1.2 UE에서의 자율주행 연구 옵션

| 접근법 | 설명 | 장점 | 단점 |
|--------|------|------|------|
| **UnrealMLAgents** | ML-Agents 포트로 PPO 학습 | Unity ML-Agents와 동일한 워크플로 | 자율주행 특화 기능 부족 |
| **Schola + Python** | Gym 인터페이스로 SB3 학습 | 유연한 알고리즘 선택 | 추가 설정 필요 |
| **CARLA** | UE4 기반 오픈소스 자율주행 시뮬레이터 | 가장 성숙, 센서/맵 풍부 | UE5 마이그레이션 제한적 |
| **AirSim** | Microsoft의 UE4/5 드론/자동차 시뮬레이터 | 현실적인 물리/센서 | 유지보수 중단 위험 |

### 5.1.3 UnrealMLAgents로 자율주행 에이전트 구현

```cpp
// CarAgent.h
#pragma once

#include "CoreMinimal.h"
#include "Agents/AgentBase.h"
#include "WheeledVehiclePawn.h"
#include "CarAgent.generated.h"

UCLASS()
class MYPROJECT_API ACarAgent : public AAgentBase
{
    GENERATED_BODY()

public:
    UPROPERTY(EditAnywhere, Category = "Car")
    float MaxSpeed = 5000.0f;

    UPROPERTY(EditAnywhere, Category = "Sensors")
    int32 NumRaycasts = 5;

    UPROPERTY(EditAnywhere, Category = "Sensors")
    float RaycastLength = 2000.0f;

    virtual void CollectObservations(FObservationBuffer& Buffer) override;
    virtual void OnActionReceived(const FActionBuffer& Actions) override;
    virtual void OnEpisodeBegin() override;

    virtual void Tick(float DeltaTime) override;

private:
    float CurrentSpeed = 0.0f;
    float LapProgress = 0.0f;
    TArray<float> RayDistances;
    UPROPERTY()
    class AWheeledVehicleMovementComponent* VehicleMovement;
    UPROPERTY()
    class UBoxComponent* CheckpointTrigger;
};
```

```cpp
// CarAgent.cpp
#include "CarAgent.h"
#include "Agents/ObservationBuffer.h"
#include "Agents/ActionBuffer.h"
#include "WheeledVehicleMovementComponent.h"
#include "Components/BoxComponent.h"

void ACarAgent::OnEpisodeBegin()
{
    // 시작 위치로 리셋
    SetActorLocation(StartTransform.GetLocation());
    SetActorRotation(StartTransform.GetRotation());

    if (VehicleMovement)
    {
        VehicleMovement->SetThrottleInput(0.0f);
        VehicleMovement->SetSteeringInput(0.0f);
        VehicleMovement->SetBrakeInput(1.0f);
    }
    CurrentSpeed = 0.0f;
    LapProgress = 0.0f;
}

void ACarAgent::CollectObservations(FObservationBuffer& Buffer)
{
    // 속도 정보
    CurrentSpeed = GetVelocity().Size();
    Buffer.AddObservation(CurrentSpeed / MaxSpeed);

    // 전방 속도
    float ForwardSpeed = FVector::DotProduct(GetVelocity(), GetActorForwardVector());
    Buffer.AddObservation(ForwardSpeed / MaxSpeed);

    // 측면 속도
    float LateralSpeed = FVector::DotProduct(GetVelocity(), GetActorRightVector());
    Buffer.AddObservation(LateralSpeed / MaxSpeed);

    // Ray Perception (5개 Ray, 전방 180도 분포)
    TArray<float> RayAngles = {-90, -45, 0, 45, 90};
    FVector Start = GetActorLocation();
    FCollisionQueryParams Params;
    Params.AddIgnoredActor(this);

    for (float Angle : RayAngles)
    {
        FRotator RayRot = GetActorRotation() + FRotator(0, Angle, 0);
        FVector End = Start + RayRot.Vector() * RaycastLength;

        FHitResult Hit;
        if (GetWorld()->LineTraceSingleByChannel(Hit, Start, End, ECC_WorldStatic, Params))
        {
            Buffer.AddObservation(Hit.Distance / RaycastLength);  // 정규화
        }
        else
        {
            Buffer.AddObservation(1.0f);  // 최대값
        }
    }
}

void ACarAgent::OnActionReceived(const FActionBuffer& Actions)
{
    if (!VehicleMovement) return;

    // 연속 액션: [조향, 가속, 브레이크]
    float Steering = FMath::Clamp(Actions.GetContinuousAction(0), -1.0f, 1.0f);
    float Throttle = FMath::Clamp(Actions.GetContinuousAction(1), 0.0f, 1.0f);
    float Brake = FMath::Clamp(Actions.GetContinuousAction(2), 0.0f, 1.0f);

    VehicleMovement->SetSteeringInput(Steering);
    VehicleMovement->SetThrottleInput(Throttle);
    VehicleMovement->SetBrakeInput(Brake);

    // 전진 보상
    float ForwardSpeed = FVector::DotProduct(GetVelocity(), GetActorForwardVector());
    AddReward(ForwardSpeed * 0.0001f * GetWorld()->GetDeltaSeconds());

    // 측면 이동 패널티
    float LateralSpeed = FVector::DotProduct(GetVelocity(), GetActorRightVector());
    AddReward(-FMath::Abs(LateralSpeed) * 0.0002f * GetWorld()->GetDeltaSeconds());
}

void ACarAgent::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);
    // 충돌 감지 등 Tick 로직
}
```

### 5.1.4 UEWheeledVehiclePawn을 사용한 차량 설정

1. `WheeledVehiclePawn` 상속 클래스 생성
2. `WheeledVehicleMovementComponent` 설정 (4륜, 전륜 조향, 후륜 구동)
3. 물리 에셋 설정 (서스펜션, 타이어 마찰력)
4. Raycast 센서 부착 (전방, 측면)
5. 체크포인트/트랙 시스템 구현

### 5.1.5 YAML 설정

```yaml
behaviors:
  CarAgent:
    trainer_type: ppo
    hyperparameters:
      batch_size: 1024
      buffer_size: 10240
      learning_rate: 3.0e-4
      beta: 0.01
      epsilon: 0.2
      lambd: 0.95
      num_epoch: 3
    network_settings:
      normalize: true
      hidden_units: 256
      num_layers: 3
    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0
    max_steps: 2.0e7
    time_horizon: 1000
    summary_freq: 10000
```

### 5.1.6 CARLA 연동 (대안)

만약 본격적인 자율주행 연구가 목적이라면, **CARLA** 시뮬레이터를 사용하는 것을 권장합니다:

```python
# CARLA + SB3 예시
import carla
import gym
import numpy as np
from stable_baselines3 import PPO

class CarlaEnv(gym.Env):
    """CARLA Gym 환경"""
    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.world = self.client.get_world()
        self.vehicle = None
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(256, 256, 3))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))

    def step(self, action):
        throttle = float((action[0] + 1) / 2)
        steer = float(action[1])
        self.vehicle.apply_control(carla.VehicleControl(
            throttle=throttle, steer=steer
        ))
        # ... observation, reward, done 처리
        return obs, reward, done, {}

    def reset(self):
        # 환경 리셋
        return obs
```

### 5.1.7 자율주행 학습 팁 (UE)

1. **멀티 환경**: 여러 UE 인스턴스를 실행하여 학습 속도 향상 (`--num-envs=4`)
2. **체크포인트 보상**: 주행 거리보다 체크포인트 통과 보상이 효과적
3. **커리큘럼**: 직선 → 곡선 → 장애물 → 교차로 순난이도 증가
4. **센서 융합**: Ray Perception + 카메라(시각) + 속도계 결합
5. **Sim-to-Real**: UE의 현실적인 물리/렌더링으로 실제 환경 전이 학습 가능

---

## 5.2 Unreal ML 유튜브 및 커뮤니티 사례

### 5.2.1 주요 YouTube 채널 및 튜토리얼

#### 1. UnrealMLAgents 공식 채널 (AlanLaboratory)
- **GitHub**: https://github.com/AlanLaboratory/UnrealMLAgents
- **내용**: 플러그인 설치, 기본 에이전트 생성, Python 트레이너 연동
- **특징**: Unity ML-Agents 사용자를 위한 마이그레이션 가이드

#### 2. Schola 튜토리얼 (GPUOpen)
- **GitHub**: https://github.com/GPUOpen-LibrariesAndSDKs/Schola
- **내용**: AMD GPUOpen 공식 튜토리얼, Python Gym 연동
- **특징**: 연구 중심, 하이퍼파라미터 튜닝 가이드

#### 3. Unreal Engine 공식 문서 및 포럼
- **UE5 Learning Agents**: https://dev.epicgames.com/documentation/
- **내용**: 에픽게임즈 공식 실험 플러그인 문서
- **한계**: 문서가 매우 제한적, 샘플 코드 부족

#### 4. 커뮤니티 튜토리얼
- **UE ML-Agents 예제**: UE 마켓플레이스/Fab에서 플러그인 구매 후 제공되는 예제
- **자율주행 예제**: UE 기반 자율주행 연구 (개인 프로젝트)
- **로보틱스 시뮬레이션**: UE + ROS 연동 튜토리얼

### 5.2.2 학습 자료 추천 순서

| 단계 | 리소스 | 설명 |
|------|--------|------|
| 1 | UnrealMLAgents GitHub README | 설치 및 기본 예제 |
| 2 | Schola 튜토리얼 (GPUOpen) | Gym 인터페이스 익히기 |
| 3 | UE5 C++ 기본 문서 | UE C++ 기초 (액터, 컴포넌트, 물리) |
| 4 | SB3 문서 (stable-baselines3.readthedocs.io) | Python RL 알고리즘 이해 |
| 5 | CARLA/예제 프로젝트 | 실제 연구/게임 적용 사례 |

### 5.2.3 참고할 만한 오픈소스 프로젝트

| 프로젝트 | 설명 | 링크 |
|----------|------|------|
| **UnrealMLAgents** | Unity ML-Agents의 UE 포트 | https://github.com/AlanLaboratory/UnrealMLAgents |
| **Schola** | AMD GPUOpen RL 플러그인 | https://github.com/GPUOpen-LibrariesAndSDKs/Schola |
| **ue-la-example** | UE5 Learning Agents 예제 | https://github.com/automathan/ue-la-example |
| **ue_learning_agents_tutorial** | Learning Agents 튜토리얼 시리즈 | https://github.com/gensen/ue_learning_agents_tutorial |
| **Unity-Robotics-Hub** | 참고: Unity 로보틱스 (UE에도 동일 개념 적용) | https://github.com/Unity-Technologies/Unity-Robotics-Hub |

---

## 5.3 산업 문제에 UE ML 적용 사례

### 5.3.1 로보틱스 (Robotics)

**사례: UE + ROS2 연동 로봇 시뮬레이션**

| 구성 요소 | 설명 |
|-----------|------|
| **UE5** | 고품질 3D 시뮬레이션 환경 |
| **ROS2** | 로봇 제어 미들웨어 |
| **Schola/UnrealMLAgents** | ML 학습 인터페이스 |
| **Python** | PPO/SAC 트레이너 |

**UE-ROS2 연동 방법:**
- `rosbridge_suite`를 통한 WebSocket 통신
- UE4/5의 `ROSIntegration` 플러그인
- 커스텀 TCP 소켓 통신

### 5.3.2 물류 및 창고 자동화

**사례: 다중 로봇 창고 관리**
- Schola의 Gym 인터페이스를 사용한 다중 에이전트 학습
- 각 로봇이 협력하여 물품 분류 및 운반
- 충돌 회피, 경로 최적화, 작업 분배 학습
- UE의 현실적인 물리 엔진으로 실제 환경 시뮬레이션

### 5.3.3 제조 공정 최적화

**사례: 조립 라인 시뮬레이션**
- UE5의 `Physics Constraints`를 사용한 조립 공정 모델링
- 에이전트가 로봇 팔을 제어하여 부품 조립
- Curriculum Learning으로 단계별 난이도 증가
- 실제 생산 데이터 기반 시뮬레이션

### 5.3.4 건축 및 도시 시뮬레이션

**사례: 군중 시뮬레이션**
- UE의 대규모 환경에서 다중 에이전트 군중 행동 학습
- 비상 상황 대피 경로 최적화
- 보행자 흐름 분석 및 혼잡도 예측
- 실제 도시 레이아웃 기반 시뮬레이션

### 5.3.5 의료 시뮬레이션

**사례: 수술 로봇 훈련**
- UE5의 고품질 렌더링으로 실제 수술 환경 재현
- Schola를 통한 로봇 팔 제어 학습
- 안전한 가상 환경에서의 반복 훈련
- 실제 환자 데이터 기반 시나리오

---

## 5.4 상용 게임에 UE ML 적용 사례

### 5.4.1 게임 AI 테스트 및 QA

**사례: 자동화된 게임 테스트**
- UnrealMLAgents를 사용한 버그 탐지 에이전트
- UE의 `GameplayDebugger`와 연동한 행동 로깅
- 다양한 게임 시나리오 자동 탐색
- 회귀 테스트 자동화

```cpp
// UE AI 테스트 에이전트 예시
void ATestAgent::CollectObservations(FObservationBuffer& Buffer)
{
    // 게임 상태 관측
    Buffer.AddObservation(GetHealth() / 100.0f);
    Buffer.AddObservation(GetAmmo() / 30.0f);
    Buffer.AddObservation(GetWorld()->GetTimeSeconds() / 300.0f);

    // 레벨 탐색 진행도
    Buffer.AddObservation(ExplorationProgress);
}
```

### 5.4.2 NPC 행동 제어

**사례: 지능형 적 AI (FPS/TPS)**
- UnrealMLAgents/Schola로 훈련된 적 캐릭터
- UE `BehaviorTree` + ML 혼합 접근법
  - BehaviorTree: 기본 전술 (엄폐, 이동 패턴)
  - ML: 플레이어 적응형 난이도 조절, 전략 결정
- 팀워크를 갖춘 적 그룹 행동

```cpp
// BehaviorTree와 ML 에이전트 혼합
// 1. BehaviorTree가 기본 상태 관리
// 2. ML 에이전트가 결정이 필요한 순간에 행동 선택
// 3. 결과를 BehaviorTree의 Blackboard에 반영

void AMLEnemyController::BeginPlay()
{
    Super::BeginPlay();
    // ML 에이전트 초기화
    MLAgent = NewObject<UMLAgentComponent>(this);
    MLAgent->Initialize(ObservationSize, ActionSize);
}

void AMLEnemyController::OnDecisionRequest()
{
    // BehaviorTree에서 ML 결정 요청 시
    float Decision = MLAgent->GetDecision(CurrentObservation);
    Blackboard->SetValueAsFloat("MLDecision", Decision);
}
```

### 5.4.3 게임 디자인 평가

**사례: 레벨 난이도 테스트**
- ML 에이전트가 레벨을 플레이하며 난이도 평가
- UE `Automation` 시스템과 연동
- 데이터 기반 게임 디자인 의사 결정
- 다양한 레이아웃 자동 테스트

### 5.4.4 UE Network 최적화 (추론)

UE에서 학습된 ML 모델을 상용 게임에 배포할 때 고려사항:

```cpp
// UE에서 ONNX 모델 추론 (DirectML/ONNX Runtime)
#include "OnnxRuntime/OnnxRuntime.h"

UCLASS()
class UGameAIModel : public UObject
{
    GENERATED_BODY()

public:
    UPROPERTY()
    UOnnxModel* Model;

    void LoadModel(const FString& ModelPath)
    {
        Model = NewObject<UOnnxModel>(this);
        Model->Load(ModelPath);
    }

    TArray<float> Infer(const TArray<float>& Input)
    {
        TArray<float> Output;
        Model->Run(Input, Output);
        return Output;
    }
};
```

**성능 최적화 팁:**

| 방법 | 설명 | 효과 |
|------|------|------|
| ONNX Runtime | UE 플러그인으로 ONNX 모델 직접 실행 | Python 통신 불필요 |
| DirectML | DirectX 12 기반 GPU 추론 | NVIDIA/AMD GPU 가속 |
| 모델 양자화 | FP32 → FP16/INT8 변환 | 속도 2~4배 향상 |
| 배치 처리 | 여러 에이전트 동시 추론 | CPU 활용도 향상 |
| LOD 시스템 | 중요도에 따라 모델 정확도 차등 적용 | 프레임 영향 최소화 |

### 5.4.5 상용 게임 적용 팁 (UE)

1. **Inference Only 모드**: 출시 시 Python 연결 없이 ONNX 모델로만 동작
2. **BehaviorTree 혼합**: 복잡한 게임 AI는 BehaviorTree + ML 혼합 사용
3. **모델 최적화**:
   - `hidden_units` 최소화 (성능 최적화)
   - ONNX Runtime 또는 DirectML 사용
   - FP16/INT8 양자화
4. **비동기 추론**: 게임 루프를 블로킹하지 않도록 별도 스레드에서 추론
5. **디버깅**: UE `Visual Logger`로 ML 결정 과정 기록

```
[ML-Agents 개발 → 출시 워크플로우]

개발 단계:          UE (Play) ◄──gRPC──► Python (학습)
                    (Behavior Type: Default)

출시 단계:          UE (Standalone)
                    └── ONNX 모델 (Behavior Type: Inference Only)
                    └── BehaviorTree + ML 혼합

성능 모니터링:      UE Insights / Unreal Insights
                    └── ML 추론 시간 추적
                    └── 프레임 타임에 미치는 영향 분석
```

### 5.4.6 실제 적용 커뮤니티 사례

UE + ML이 적용된 커뮤니티 프로젝트:
- **ML 전투 로봇**: UnrealMLAgents로 전투 로봇 AI 학습
- **자율주행 레이싱**: UE5의 Chaos Physics 기반 레이싱 AI
- **군집 비행**: Schola 다중 에이전트로 드론 군집 제어
- **대화형 NPC**: ML 기반 감정/행동 시스템을 갖춘 NPC

> **참고**: UE ML 생태계는 Unity ML-Agents에 비해 초기 단계입니다. 대부분의 프로젝트는 GitHub/Fab에서 확인 가능하며, 커뮤니티가 성장 중입니다.

---

# 부록

## A. 유용한 명령어 모음

```bash
# ==========================================
# Conda 환경 관리
# ==========================================

# 환경 생성 및 활성화
conda create -n ue_ml python=3.10 -y
conda activate ue_ml

# environment.yml로 환경 생성 (재현 가능)
conda env create -f environment_ue_ml_full.yml
conda activate ue_ml_full

# 환경 업데이트 (yml 파일 변경 시)
conda env update -f environment_ue_ml_full.yml --prune

# 환경 내보내기 (재현용)
conda env export -n ue_ml_full > environment_backup.yml          # 전체 (플랫폼 종속)
conda env export -n ue_ml_full --from-history > environment.yml  # 간소화 (이식성 높음)

# 환경 복제
conda create -n ue_ml_backup --clone ue_ml

# 환경 목록 보기
conda info --envs
conda env list

# 환경 제거
conda deactivate
conda env remove -n ue_ml_full

# Conda 자체 관리
conda update conda -n base -c conda-forge
conda clean --all           # 캐시/임시 파일 전체 삭제
conda clean -p              # 미사용 패키지 삭제
conda clean -t              # tarball 캐시 삭제

# Conda 정보
conda info
conda list -n ue_ml         # 특정 환경의 패키지 목록

# Mamba (더 빠른 conda 대안) 설치 및 사용
conda install mamba -n base -c conda-forge
mamba env create -f environment_ue_ml_full.yml    # conda보다 3~5배 빠름
mamba install pytorch=2.1.0 cudatoolkit=11.8 -c pytorch -c nvidia

# ==========================================
# UnrealMLAgents 학습 실행
# ==========================================
mlagents-learn <config> --run-id=<id> [--num-envs=N] [--resume] [--force]

# ==========================================
# Schola 학습 (Python 직접 실행)
# ==========================================
python train_schola.py

# ==========================================
# TensorBoard 모니터링
# ==========================================
tensorboard --logdir=results
tensorboard --logdir=./tb_logs/

# ==========================================
# UE5 소스 빌드 (Learning Agents 필요 시)
# ==========================================
Setup.bat && GenerateProjectFiles.bat

# ==========================================
# Python 환경 확인
# ==========================================
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import stable_baselines3; print(stable_baselines3.__version__)"
python -c "import gymnasium; print(gymnasium.__version__)"

# ==========================================
# ONNX 모델 확인
# ==========================================
python -c "import onnx; model = onnx.load('model.onnx'); onnx.checker.check_model(model)"

# ==========================================
# 설치된 패키지 확인
# ==========================================
pip list | grep -E "mlagents|schola|torch|stable|gym"
conda list -n ue_ml | findstr -E "torch|cudatoolkit"   # Windows
conda list -n ue_ml | grep -E "torch|cudatoolkit"      # Linux/macOS

## B. 참고 자료

| 자료 | 링크 |
|------|------|
| UnrealMLAgents GitHub | https://github.com/AlanLaboratory/UnrealMLAgents |
| Schola (GPUOpen) | https://github.com/GPUOpen-LibrariesAndSDKs/Schola |
| UE5 Learning Agents 예제 | https://github.com/automathan/ue-la-example |
| UE Learning Agents 튜토리얼 | https://github.com/gensen/ue_learning_agents_tutorial |
| UE5 공식 문서 | https://dev.epicgames.com/documentation/ |
| Stable-Baselines3 | https://stable-baselines3.readthedocs.io/ |
| Unity ML-Agents (참고) | https://github.com/Unity-Technologies/ml-agents |
| CARLA 시뮬레이터 | https://carla.org/ |

## C. 하이퍼파라미터 퀵 레퍼런스

### UnrealMLAgents (Unity ML-Agents 포맷)

```
[간단한 환경 - Dodge, PushBlock]
batch_size: 128 | buffer_size: 2048 | hidden_units: 128 | num_layers: 2
max_steps: 5e5 | time_horizon: 64

[중간 환경 - Maze, 자율주행]
batch_size: 1024 | buffer_size: 10240 | hidden_units: 256 | num_layers: 3
max_steps: 5e6 | time_horizon: 256

[복잡한 환경 - 멀티 에이전트, 대규모]
batch_size: 4096 | buffer_size: 65536 | hidden_units: 512 | num_layers: 3~4
max_steps: 1e7+ | time_horizon: 1000
```

### SB3 직접 사용 시

```python
# 간단한 환경
PPO("MlpPolicy", env, n_steps=2048, batch_size=64, n_epochs=10, net_arch=[64, 64])

# 중간 환경
PPO("MlpPolicy", env, n_steps=2048, batch_size=256, n_epochs=10, net_arch=[256, 256])

# 복잡한 환경
PPO("MlpPolicy", env, n_steps=4096, batch_size=1024, n_epochs=5, net_arch=[512, 512])
```

## D. Unity ML-Agents vs Unreal ML 맵핑

| Unity ML-Agents 개념 | Unreal ML 대응 |
|-----------------------|----------------|
| Agent C# 클래스 | AgentBase C++ (UnrealMLAgents) / ScholaEnv (Schola) |
| Behavior Parameters | UA_BehaviorParameters 컴포넌트 |
| Decision Requester | UA_DecisionRequester 컴포넌트 |
| Ray Perception Sensor | UE LineTrace / Raycast 직접 구현 |
| Buffer Sensor (가변 길이) | UE TArray + 패딩 방식 |
| SimpleMultiAgentGroup | 수동 그룹 관리 (C++ / Python) |
| PPO/SAC 내장 트레이너 | SB3 Python (외부) |
| Ghost Trainer (Self-Play) | Python 수동 구현 |
| MA-POCA | Python 그룹 보상 구현 |
| RND | SB3 콜백 구현 |
| Curriculum Learning | Python 환경 파라미터 제어 |
| ONNX 모델 | ONNX Runtime / DirectML |
| gRPC 통신 | UnrealMLAgents (gRPC 호환) / Schola (TCP Socket) |
| Unity Package Manager | UE 플러그인 시스템 (Plugins 폴더) |
| mlagents-learn CLI | 동일 (UnrealMLAgents 호환) |
| **Python 환경 관리** | venv (내장) 또는 Conda (Miniconda) + environment.yml |
| **requirements.txt** | `environment.yml` (conda) 또는 `requirements.txt` (pip) |
| **CUDA 버전 관리** | `cudatoolkit` conda 패키지로 버전 고정 |

---

> **문서 작성일**: 2026년 6월 9일  
> **Unreal Engine 버전**: 5.6 기준  
> **주요 툴킷**: UnrealMLAgents v0.0.9, Schola v2.0, UE5 Learning Agents (Experimental)  
> **참고**: Unreal ML 생태계는 Unity ML-Agents에 비해 성숙도가 낮으며, 지속적으로 발전 중입니다.

# Pipe Flange Inspection — MuJoCo Simulation

6-DOF 로봇 팔이 TCP에 장착된 D405 카메라로 파이프 플랜지 용접부를 검사하는 MuJoCo 시뮬레이션.

---

## 환경 설정

**Python 3.11** 필요.

```bash
# 가상환경 생성 (최초 1회)
python3.11 -m venv ~/.venvs/we_meet

# 가상환경 활성화
source ~/.venvs/we_meet/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

---

## 실행

### 기본 실행

```bash
python main.py
```

실행하면 세 개의 창이 뜹니다.

- **MuJoCo 뷰어** — 로봇 3D 시뮬레이션
- **대시보드** — 관절값(q), 기준값(q_ref), 에러(e_q) 실시간 그래프
- **3D 스켈레톤** — 로봇 뼈대 + 기준 궤적 + 카메라 위치

> **macOS 주의**: MuJoCo 뷰어가 포함된 명령은 `python` 대신 `mjpython`을 사용해야 합니다.
> ```bash
> mjpython main.py  # macOS
> python main.py    # Ubuntu
> ```

### 카메라 뎁스 이미지 저장

RGB 컬러맵 PNG와 `.npy` 뎁스 배열을 `inspection_frames/`에 저장합니다.

```bash
python mujoco_viewer.py --camera --no-viewer
```

### 그 외 자주 쓰는 명령어

```bash
# 헤드리스 녹화 — out/sim.mp4 + out/log.csv + out/summary.png 저장
python main.py --record

# IK 검증만 (뷰어 없이 빠르게 확인)
python mujoco_viewer.py --ik --verify --no-viewer

# 포즈·관절 CSV 내보내기
python mujoco_viewer.py --ik --export-csv --no-viewer

# 에러 히트맵 추가
python main.py --heatmap

# 3D 스켈레톤 창 끄기
python main.py --no-skeleton
```

---

## 출력 파일

| 경로 | 내용 | 생성 명령 |
|------|------|-----------|
| `inspection_frames/depth_png/frame_*.png` | 뎁스 시각화 PNG | `--camera` |
| `inspection_frames/depth_meters/frame_*.npy` | 미터 단위 뎁스 배열 | `--camera` |
| `inspection_frames/metadata.csv` | 프레임별 포즈·관절·유효성 메타 | `--export-csv` |
| `out/log.csv` | 시뮬레이션 전체 샘플 (t, q, q_ref, p_tcp …) | `--live` / `--record` |
| `out/sim.mp4` | 오프스크린 렌더 영상 | `--record` |
| `out/summary.png` | 관절 오차·RMSE 요약 플롯 | `--record` |

---

## 프로젝트 구조

```
we_meet/
├── main.py                      # 라이브 시뮬레이션 엔트리포인트
├── mujoco_viewer.py             # IK + 카메라 렌더링 뷰어
├── scene.xml                    # MuJoCo 전체 씬 (로봇 + 파이프 + 플랜지)
├── robot_model.xml              # 6-DOF DH 로봇 단독 모델
├── requirements.txt
├── control/
│   ├── franka_ik_solver.py      # MuJoCo Jacobian IK
│   └── clik.py
├── trajectory/
│   └── generator.py            # 플랜지 검사 궤적 생성
├── viz/
│   ├── combined_view.py        # 대시보드 + 3D 스켈레톤 통합 창
│   ├── dashboard.py
│   └── skeleton3d.py
├── tools/
│   ├── export_inspection_dataset.py
│   └── visualize_frames.py
├── inspection_frames/           # 카메라 뎁스 출력 (자동 생성)
└── out/                         # 시뮬레이션 로그·영상 출력 (자동 생성)
```

---

## 주요 파라미터

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `--retries` | 16 | IK 랜덤 재시도 횟수 |
| `--playback-speed` | 1.0 | 시뮬레이션 재생 속도 배율 |
| `--out-dir` | `out/` | 로그·영상 저장 경로 |
| `--camera-name` | `d405_camera` | 렌더링할 MuJoCo 카메라 이름 |
| `--mujoco-waypoints` | 240 | IK 웨이포인트 수 |

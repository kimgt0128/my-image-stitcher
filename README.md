# my-image-stitcher
# video-panorama-stitching

동영상에서 프레임을 추출한 뒤, 여러 프레임을 자동으로 정합하여 하나의 넓은 파노라마 이미지를 생성하는 프로그램입니다.

OpenCV의 Stitcher를 사용하지 않고  
특징점 → 매칭 → RANSAC → 투영 → 블렌딩 → 크롭까지 전 과정을 직접 구현했습니다.

---

## 데모

### GIF 생성

~~~bash
brew install ffmpeg
mkdir -p screenshots

ffmpeg -i input_video/input.mp4 -t 5 -vf "fps=10,scale=640:-1" screenshots/input.gif
~~~

---

## 결과 비교

<table>
  <tr>
    <th>Input</th>
    <th>v1</th>
  </tr>
  <tr>
    <td><img src="screenshots/input.gif" width="420"/></td>
    <td><img src="v1/panorama_improved.jpg" width="420"/></td>
  </tr>
</table>

<br>

<table>
  <tr>
    <th>v2</th>
    <th>v3</th>
  </tr>
  <tr>
    <td><img src="v2/panorama_improved_v2.jpg" width="420"/></td>
    <td><img src="v3/panorama_improved_v3.jpg" width="420"/></td>
  </tr>
</table>

---

## 전체 파이프라인

~~~text
Video
 → Frame Extraction
 → Cylindrical Warping
 → Feature Matching
 → Motion Estimation
 → Image Placement
 → Blending
 → Crop
~~~

---

# v1 — SIFT + Homography

## 핵심

- SIFT 특징점
- FLANN 매칭
- RANSAC Homography
- 중앙 기준 정렬
- Gaussian blending

## 문제

- ghost 발생
- 경계선 보임

---

# v2 — Blending 개선

## 추가

- distance 기반 weight
- feather blending 개선
- auto crop 개선

## 효과

- 경계 부드러움 증가
- 검은 영역 감소

---

# v3 — 구조 개선 (핵심)

## 핵심 변화

~~~text
Homography → Translation
~~~

## 이유

~~~text
원통형 투영 이후 = 거의 수평 이동 문제
~~~

---

## 주요 기능

- ORB 특징점
- RANSAC translation
- keyframe 선택
- seam-cut blending
- 색상 보정

---

## 핵심 아이디어

### Keyframe 선택

~~~text
프레임 많음 → ghost 증가
프레임 적음 → 끊김

→ 적절한 간격 선택
~~~

---

### Seam-cut blending

~~~text
기존: 전체 평균 → 흐림
개선: 경계만 blending → 선명
~~~

---

### 색상 보정

~~~text
Lab 공간에서 평균 맞춤
→ 색상 일관성 개선
~~~

---

# 실험 결과

| ratio | 결과 |
|------|------|
| 0.70 | 번짐 |
| 0.78 | 가장 안정 |
| 0.82 | 끊김 |

---

# 트러블슈팅

## 1. 프레임 많으면 좋은가?

❌ 아니다

~~~text
많으면 → overlap 많음 → ghost 증가
~~~

✔ 해결: keyframe 선택

---

## 2. 회전 보정 실패

~~~text
Affine 적용 → 정렬 깨짐
~~~

✔ 해결: 제거

---

## 3. ghost 완전 제거 불가

### 이유

~~~text
깊이 차이 (parallax)
~~~

- 가까운 물체
- 먼 물체

→ 하나의 transform으로 해결 불가

---

# 추가 구현 사항

| 기능 | 적용 |
|------|------|
| cylindrical view | O |
| blending | O |
| crop | O |
| keyframe | O |
| seam blending | O |
| color 보정 | O |

---

# v4 확장 아이디어

## 1. seam finding

~~~text
최소 차이 경로 탐색
~~~

---

## 2. multi-band blending

~~~text
저주파/고주파 분리
~~~

---

## 3. mesh warping

~~~text
지역별 변환
~~~

---

## 4. optical flow

~~~text
dense motion 적용
~~~

---

# 실행 방법

~~~bash
python v1/stitch.py
python v2/stitch2.py
python v3/stitch3.py
~~~

---

# 옵션

~~~bash
python v3/stitch3.py --frames 50 --scale 0.5
~~~

---

# 폴더 구조

~~~text
.
├── README.md
├── input_video
│   └── input.mp4
├── screenshots
│   └── input.gif
├── v1
├── v2
└── v3
~~~

---

# 요구사항

~~~bash
pip install opencv-python numpy
~~~

---

# 최종 정리

~~~text
v1 → 기본 구현
v2 → blending 개선
v3 → 구조 개선 (핵심)

→ 점진적 개선 과정이 핵심
~~~
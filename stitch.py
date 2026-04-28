"""
stitch.py — Image Stitching (HW5, SEOULTECH Computer Vision)

Pipeline:
  1. Extract N frames from video (or load images directly)
  2. SIFT feature detection + descriptor extraction
  3. BruteForce-L2 matching + Lowe's ratio test (threshold=0.75)
  4. RANSAC Homography estimation (cv.findHomography, threshold=4.0)
  5. Center-based cumulative Homography composition
  6. Vertical drift correction (flatten staircase artifact)
  7. Distance-weighted alpha blending at seams
  8. Auto-crop black borders
  9. Save panorama.jpg

Usage:
  python stitch.py --video input.mp4 --frames 12 --start 2 --end 18
  python stitch.py --images img1.jpg img2.jpg img3.jpg
"""

import argparse
import sys
import cv2 as cv
import numpy as np


# ──────────────────────────────────────────
# 1. 프레임 추출
# ──────────────────────────────────────────

def extract_frames(video_path, n_frames, start_sec=2.0, end_sec=18.0, rotate=0):
    """
    비디오에서 [start_sec, end_sec] 구간을 n_frames 개로 균등 추출.
    rotate: 0 / 90 / -90 / 180 (도)
    """
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'[ERROR] Cannot open video: {video_path}')
        sys.exit(1)

    fps         = cap.get(cv.CAP_PROP_FPS)
    total_f     = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    start_frame = int(start_sec * fps)
    end_frame   = min(int(end_sec * fps), total_f - 1)

    indices = np.linspace(start_frame, end_frame, n_frames, dtype=int)

    rot_map = {
        90:  cv.ROTATE_90_CLOCKWISE,
        -90: cv.ROTATE_90_COUNTERCLOCKWISE,
        270: cv.ROTATE_90_COUNTERCLOCKWISE,
        180: cv.ROTATE_180,
    }

    frames = []
    for idx in indices:
        cap.set(cv.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            print(f'[WARNING] Failed to read frame {idx}, skipping')
            continue
        if rotate in rot_map:
            frame = cv.rotate(frame, rot_map[rotate])
        frames.append(frame)

    cap.release()
    print(f'[INFO] Loaded {len(frames)} frames  '
          f'(t={start_sec:.1f}s ~ {end_sec:.1f}s,  fps={fps:.1f})')
    return frames


# ──────────────────────────────────────────
# 2. 특징점 매칭
# ──────────────────────────────────────────

def detect_and_match(img1, img2, ratio_thresh=0.75):
    """
    SIFT 검출 + BruteForce-L2 + Lowe's ratio test.
    반환: pts1, pts2 (float32 배열), n_good (매칭 수)
    """
    sift = cv.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)

    if desc1 is None or desc2 is None:
        return np.array([]), np.array([]), 0

    matcher     = cv.BFMatcher(cv.NORM_L2)
    raw_matches = matcher.knnMatch(desc1, desc2, k=2)

    good = [m for m, n in raw_matches if m.distance < ratio_thresh * n.distance]

    if len(good) == 0:
        return np.array([]), np.array([]), 0

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return pts1, pts2, len(good)


# ──────────────────────────────────────────
# 3. Homography 추정
# ──────────────────────────────────────────

def compute_homography(pts1, pts2, ransac_thresh=4.0):
    """
    pts2 → pts1 변환 Homography를 RANSAC으로 추정.

    순수 회전 카메라에서 H = K·R·K⁻¹ 이므로 perspective 항(H[2,0], H[2,1])은
    이론적으로 0이어야 함. RANSAC 노이즈로 생긴 작은 값이 누적 합성 시 기하급수적으로
    왜곡을 키우므로, 추정 후 perspective 행을 강제로 0으로 설정.

    반환: H (3×3, perspective-free), n_inliers
    """
    if len(pts1) < 4:
        return None, 0

    H, mask = cv.findHomography(pts2, pts1, cv.RANSAC, ransac_thresh)
    if H is None:
        return None, 0

    # perspective 항 제거: 순수 회전 panorama에서는 0이어야 함
    H[2, 0] = 0.0
    H[2, 1] = 0.0
    H[2, 2] = 1.0

    n_inliers = int(mask.sum()) if mask is not None else 0
    return H, n_inliers


# ──────────────────────────────────────────
# 4. auto-crop: 검은 테두리 제거
# ──────────────────────────────────────────

def auto_crop(img):
    """유효 픽셀 bounding box로 검은 테두리 제거."""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    rows = np.any(gray > 0, axis=1)
    cols = np.any(gray > 0, axis=0)
    if not rows.any():
        return img
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return img[r0:r1 + 1, c0:c1 + 1]


# ──────────────────────────────────────────
# 5. 순차 스티칭 (누적 Homography + 알파 블렌딩)
# ──────────────────────────────────────────

def stitch_all(images):
    """
    연속된 원본 프레임끼리 pairwise 매칭 후 누적 Homography를 합성,
    전체 프레임을 공통 캔버스에 한 번에 워핑.

    개선사항:
    - 세로 드리프트 보정: 모든 프레임 중심의 canvas y 좌표 평균을 맞춰
      수평 팬닝 중 발생한 미세 카메라 틸트로 인한 계단 현상을 완화.
    - 거리 가중 알파 블렌딩: 각 프레임의 경계에서 중심으로의 거리를
      블렌딩 가중치로 써서 이음새를 자연스럽게 합성.
    """
    n = len(images)

    # ── Step 1: 연속 프레임 간 pairwise Homography 계산 ──
    H_pairwise = []
    for i in range(n - 1):
        label = f'{i+1}+{i+2}'
        pts1, pts2, n_matches = detect_and_match(images[i], images[i + 1])

        if n_matches < 10:
            print(f'[ERROR] Matching {label}: too few matches ({n_matches}).')
            sys.exit(1)

        H, n_inliers = compute_homography(pts1, pts2)

        if H is None or n_inliers < 10:
            print(f'[ERROR] Homography {label}: insufficient inliers ({n_inliers}).')
            sys.exit(1)

        print(f'[INFO] Stitching {label}: {n_inliers} inliers / {n_matches} matches')
        H_pairwise.append(H)

    # ── Step 2: 누적 Homography 합성 (중앙 프레임 기준) ──
    #
    # 중앙 프레임 기준으로 쓰면 최대 합성 깊이가 절반으로 줄어 왜곡 최소화.
    # H_pairwise[i]: images[i+1] 좌표 → images[i] 좌표 변환
    # H_cum[i]:      images[i]   좌표 → images[center] 좌표 변환
    center_idx = n // 2
    H_cum = [None] * n
    H_cum[center_idx] = np.eye(3, dtype=np.float64)

    for i in range(center_idx - 1, -1, -1):
        H_cum[i] = H_cum[i + 1] @ np.linalg.inv(H_pairwise[i])

    for i in range(center_idx + 1, n):
        H_cum[i] = H_cum[i - 1] @ H_pairwise[i - 1]

    print(f'[INFO] Reference frame: {center_idx + 1} (center)')

    # ── Step 3: 세로 드리프트 보정 ──
    #
    # 카메라가 수평 팬닝 중 미세하게 위아래로 기울면 각 프레임이 canvas에서
    # 다른 y 위치에 배치되어 계단 현상이 생김.
    # 모든 프레임 중심의 y 좌표 평균을 구해 H_cum의 y 오프셋을 직접 수정한 뒤
    # 캔버스 크기를 다시 계산해야 보정된 위치가 캔버스 밖으로 나가지 않음.
    y_centers = []
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        center_pt = np.float32([[w / 2, h / 2]]).reshape(1, 1, 2)
        proj = cv.perspectiveTransform(center_pt, H_cum[i]).ravel()
        y_centers.append(proj[1])

    y_mean = float(np.mean(y_centers))
    for i in range(n):
        H_cum[i][1, 2] += y_mean - y_centers[i]

    # ── Step 4: 보정 후 캔버스 크기 재계산 ──
    all_corners = []
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        warped  = cv.perspectiveTransform(corners, H_cum[i])
        all_corners.append(warped)

    all_corners = np.concatenate(all_corners, axis=0)
    x_min, y_min = all_corners.min(axis=0).ravel()
    x_max, y_max = all_corners.max(axis=0).ravel()

    offset_x = max(0.0, -x_min)
    offset_y = max(0.0, -y_min)
    canvas_w = int(np.ceil(x_max + offset_x)) + 1
    canvas_h = int(np.ceil(y_max + offset_y)) + 1

    T = np.array([[1, 0, offset_x],
                  [0, 1, offset_y],
                  [0, 0, 1       ]], dtype=np.float64)

    H_adjs = [T @ H_cum[i] for i in range(n)]

    # ── Step 5: 거리 가중 알파 블렌딩 ──
    #
    # 각 워핑된 이미지에 대해 경계에서 중심까지의 거리를 가중치로 사용.
    # 겹치는 영역에서 가중 평균을 내어 이음새를 자연스럽게 합성.
    canvas_sum = np.zeros((canvas_h, canvas_w, 3), dtype=np.float64)
    weight_sum = np.zeros((canvas_h, canvas_w),    dtype=np.float64)

    for i in range(n):
        warped = cv.warpPerspective(images[i], H_adjs[i], (canvas_w, canvas_h))
        mask   = (warped.sum(axis=2) > 0).astype(np.uint8) * 255

        # 유효 영역 내부에서 경계까지의 거리 = 블렌딩 가중치
        dist = cv.distanceTransform(mask, cv.DIST_L2, 3).astype(np.float64)

        canvas_sum += warped.astype(np.float64) * dist[:, :, np.newaxis]
        weight_sum += dist

    valid = weight_sum > 0
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[valid] = (canvas_sum[valid] / weight_sum[valid, np.newaxis]).astype(np.uint8)

    return canvas


# ──────────────────────────────────────────
# 6. 진입점
# ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Image Stitching — HW5 (SEOULTECH Computer Vision)'
    )

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--video',  type=str, default='input_video/input.mp4', help='Input video file path')
    group.add_argument('--images', nargs='+', help='Input image files (3 or more)')

    parser.add_argument('--frames', type=int,   default=12,
                        help='Frames to extract from video (default: 12)')
    parser.add_argument('--start',  type=float, default=2.0,
                        help='Video start time in seconds (default: 2.0)')
    parser.add_argument('--end',    type=float, default=18.0,
                        help='Video end time in seconds (default: 18.0)')
    parser.add_argument('--rotate', type=int,   default=0,
                        choices=[0, 90, -90, 180, 270],
                        help='Rotate each frame before stitching (default: 0)')
    parser.add_argument('--scale',  type=float, default=0.5,
                        help='Downscale factor for faster processing (default: 0.5)')
    parser.add_argument('--output', type=str,   default='panorama.jpg',
                        help='Output filename (default: panorama.jpg)')

    args = parser.parse_args()

    # ── 이미지 로드 ──
    if args.video:
        images = extract_frames(
            args.video, args.frames,
            start_sec=args.start,
            end_sec=args.end,
            rotate=args.rotate,
        )
    else:
        images = []
        for path in args.images:
            img = cv.imread(path)
            if img is None:
                print(f'[ERROR] Cannot read image: {path}')
                sys.exit(1)
            images.append(img)
        print(f'[INFO] Loaded {len(images)} images')

    if len(images) < 2:
        print('[ERROR] Need at least 2 images to stitch.')
        sys.exit(1)

    # ── 해상도 축소 (처리 속도) ──
    if args.scale != 1.0:
        images = [cv.resize(img, None, fx=args.scale, fy=args.scale,
                            interpolation=cv.INTER_AREA)
                  for img in images]
        h, w = images[0].shape[:2]
        print(f'[INFO] Scaled to {w}x{h}  (scale={args.scale})')

    # ── 스티칭 ──
    panorama = stitch_all(images)

    # ── 검은 테두리 제거 ──
    panorama = auto_crop(panorama)

    # ── 저장 ──
    cv.imwrite(args.output, panorama)
    h, w = panorama.shape[:2]
    print(f'[INFO] Saved: {args.output}  ({w}x{h})')


if __name__ == '__main__':
    main()

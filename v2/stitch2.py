import argparse
import sys
import os
import cv2 as cv
import numpy as np


# ──────────────────────────────────────────
# 1. 프레임 추출
# ──────────────────────────────────────────
def extract_frames(video_path, n_frames, start_sec=None, end_sec=None, rotate=0):
    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv.CAP_PROP_FPS)
    total_f = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or total_f <= 0:
        print("[ERROR] Invalid video information.")
        sys.exit(1)

    duration = total_f / fps

    if start_sec is None:
        start_sec = 0.0
    if end_sec is None:
        end_sec = duration

    start_frame = max(0, int(start_sec * fps))
    end_frame = min(int(end_sec * fps), total_f - 1)

    if start_frame >= end_frame:
        print("[ERROR] Invalid start/end time.")
        sys.exit(1)

    indices = np.linspace(start_frame, end_frame, n_frames, dtype=int)

    rot_map = {
        90: cv.ROTATE_90_CLOCKWISE,
        -90: cv.ROTATE_90_COUNTERCLOCKWISE,
        270: cv.ROTATE_90_COUNTERCLOCKWISE,
        180: cv.ROTATE_180
    }

    frames = []

    for idx in indices:
        cap.set(cv.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()

        if not ret:
            continue

        if rotate in rot_map:
            frame = cv.rotate(frame, rot_map[rotate])

        frames.append(frame)

    cap.release()

    print(f"[INFO] Video path: {video_path}")
    print(f"[INFO] Video duration: {duration:.2f}s")
    print(f"[INFO] Sampling from {start_sec:.2f}s to {end_sec:.2f}s")
    print(f"[INFO] Selected frame indices: {indices}")

    return frames


# ──────────────────────────────────────────
# 2. 원통형 투영
# ──────────────────────────────────────────
def cylindrical_warp(img, f):
    h, w = img.shape[:2]
    xc, yc = w / 2.0, h / 2.0

    y, x = np.indices((h, w))

    theta = (x - xc) / f
    h_cap = (y - yc) / f

    map_x = f * np.tan(theta) + xc
    map_y = f * h_cap / np.cos(theta) + yc

    mask = (
        (map_x >= 0) & (map_x < w) &
        (map_y >= 0) & (map_y < h)
    )

    warped = cv.remap(
        img,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    warped[~mask] = 0
    return warped


# ──────────────────────────────────────────
# 3. 특징점 매칭
# ──────────────────────────────────────────
def detect_and_match(img1, img2, ratio_thresh=0.7):
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create(nfeatures=4000)

    kp1, desc1 = sift.detectAndCompute(gray1, None)
    kp2, desc2 = sift.detectAndCompute(gray2, None)

    if desc1 is None or desc2 is None:
        return None, None, 0

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=80)

    matcher = cv.FlannBasedMatcher(index_params, search_params)
    raw_matches = matcher.knnMatch(desc1, desc2, k=2)

    good = []

    for pair in raw_matches:
        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < ratio_thresh * n.distance:
            good.append(m)

    if len(good) < 12:
        return None, None, len(good)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    return pts1, pts2, len(good)


def compute_homography(pts1, pts2, ransac_thresh=3.0):
    if pts1 is None or pts2 is None:
        return None, 0

    if len(pts1) < 4 or len(pts2) < 4:
        return None, 0

    H, mask = cv.findHomography(pts2, pts1, cv.RANSAC, ransac_thresh)

    if H is None or mask is None:
        return None, 0

    inliers = int(mask.sum())

    if inliers < 8:
        return None, inliers

    return H, inliers


# ──────────────────────────────────────────
# 4. 개선된 자동 crop
# ──────────────────────────────────────────
def auto_crop(img, margin_ratio=0.02):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    mask = gray > 0

    coords = np.column_stack(np.where(mask))

    if coords.size == 0:
        return img

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    cropped = img[y0:y1 + 1, x0:x1 + 1]

    h, w = cropped.shape[:2]

    margin_y = int(h * margin_ratio)
    margin_x = int(w * margin_ratio)

    y_start = margin_y
    y_end = h - margin_y
    x_start = margin_x
    x_end = w - margin_x

    if y_start >= y_end or x_start >= x_end:
        return cropped

    return cropped[y_start:y_end, x_start:x_end]


# ──────────────────────────────────────────
# 5. 개선된 feather blending weight
# ──────────────────────────────────────────
def make_smooth_alpha(mask):
    mask = mask.astype(np.float32)

    h, w = mask.shape

    dist = cv.distanceTransform((mask > 0).astype(np.uint8), cv.DIST_L2, 3)
    dist = dist.astype(np.float32)

    if dist.max() > 0:
        dist = dist / dist.max()

    x = np.linspace(0, 1, w, dtype=np.float32)
    x_weight = np.minimum(x, x[::-1])
    x_weight = x_weight / (x_weight.max() + 1e-6)
    x_weight = np.tile(x_weight, (h, 1))

    alpha = mask * dist * x_weight
    alpha = cv.GaussianBlur(alpha, (51, 51), 0)

    return alpha


# ──────────────────────────────────────────
# 6. 파노라마 스티칭
# ──────────────────────────────────────────
def stitch_all(images, focal_length):
    n = len(images)

    if n < 2:
        print("[ERROR] Need at least 2 images.")
        sys.exit(1)

    warped_images = [cylindrical_warp(img, focal_length) for img in images]

    H_pairwise = []

    for i in range(n - 1):
        pts1, pts2, n_matches = detect_and_match(
            warped_images[i],
            warped_images[i + 1]
        )

        H, n_inliers = compute_homography(pts1, pts2)

        print(f"[INFO] pair {i}-{i + 1}: matches={n_matches}, inliers={n_inliers}")

        if H is None:
            print(f"[ERROR] Matching failed between frame {i} and {i + 1}")
            print("[TIP] Try reducing --frames or increasing overlap.")
            sys.exit(1)

        H_pairwise.append(H)

    center_idx = n // 2

    H_cum = [None] * n
    H_cum[center_idx] = np.eye(3)

    for i in range(center_idx - 1, -1, -1):
        H_cum[i] = H_cum[i + 1] @ np.linalg.inv(H_pairwise[i])

    for i in range(center_idx + 1, n):
        H_cum[i] = H_cum[i - 1] @ H_pairwise[i - 1]

    all_corners = []

    for i, img in enumerate(warped_images):
        h, w = img.shape[:2]

        corners = np.float32([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ]).reshape(-1, 1, 2)

        transformed_corners = cv.perspectiveTransform(corners, H_cum[i])
        all_corners.append(transformed_corners)

    all_corners = np.concatenate(all_corners, axis=0)

    x_min, y_min = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0).ravel()).astype(int)

    canvas_w = x_max - x_min
    canvas_h = y_max - y_min

    if canvas_w <= 0 or canvas_h <= 0:
        print("[ERROR] Invalid canvas size.")
        sys.exit(1)

    T = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ], dtype=np.float64)

    canvas_sum = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight_sum = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    for i in range(n):
        H_final = T @ H_cum[i]

        warped = cv.warpPerspective(
            warped_images[i],
            H_final,
            (canvas_w, canvas_h),
            flags=cv.INTER_LINEAR,
            borderMode=cv.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        mask = (warped.sum(axis=2) > 0).astype(np.float32)
        alpha = make_smooth_alpha(mask)

        canvas_sum += warped.astype(np.float32) * alpha[..., None]
        weight_sum += alpha

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    valid = weight_sum > 1e-6

    canvas[valid] = np.clip(
        canvas_sum[valid] / weight_sum[valid, None],
        0,
        255
    ).astype(np.uint8)

    return canvas


# ──────────────────────────────────────────
# 7. main
# ──────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--start", type=float, default=None)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--rotate", type=int, default=0)
    parser.add_argument("--focal-scale", type=float, default=1.2)

    args = parser.parse_args()

    # 현재 파일: week7/v2/stitch2.py
    v2_dir = os.path.dirname(os.path.abspath(__file__))

    # week7 디렉토리
    project_dir = os.path.dirname(v2_dir)

    # 기본 입력: week7/input_video/input.mp4
    if args.video is None:
        video_path = os.path.join(project_dir, "input_video", "input.mp4")
    else:
        if os.path.isabs(args.video):
            video_path = args.video
        else:
            video_path = os.path.join(project_dir, args.video)

    # 출력: week7/v2/panorama_improved_v2.jpg
    output_path = os.path.join(v2_dir, "panorama_improved_v2.jpg")

    images = extract_frames(
        video_path,
        args.frames,
        start_sec=args.start,
        end_sec=args.end,
        rotate=args.rotate
    )

    if not images:
        print("[ERROR] No images loaded.")
        sys.exit(1)

    if args.scale != 1.0:
        images = [
            cv.resize(img, None, fx=args.scale, fy=args.scale)
            for img in images
        ]

    h, w = images[0].shape[:2]
    focal_length = w * args.focal_scale

    print(f"[INFO] Number of frames: {len(images)}")
    print(f"[INFO] Image size: {w} x {h}")
    print(f"[INFO] Estimated focal length: {focal_length}")

    panorama = stitch_all(images, focal_length)
    panorama = auto_crop(panorama)

    cv.imwrite(output_path, panorama)

    print(f"[INFO] Done! Saved as {output_path}")


if __name__ == "__main__":
    main()
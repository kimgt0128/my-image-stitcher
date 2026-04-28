import argparse
import os
import sys
import cv2 as cv
import numpy as np


# ──────────────────────────────────────────
# 1. 프레임 추출
# ──────────────────────────────────────────
def extract_frames(video_path, n_frames=50, start_sec=None, end_sec=None, rotate=0):
    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv.CAP_PROP_FPS)
    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or total <= 0:
        print("[ERROR] Invalid video information.")
        sys.exit(1)

    duration = total / fps

    if start_sec is None:
        start_sec = 0.0
    if end_sec is None:
        end_sec = duration

    start_frame = max(0, int(start_sec * fps))
    end_frame = min(total - 1, int(end_sec * fps))

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
    print(f"[INFO] Extracted frames: {len(frames)}")
    print(f"[INFO] Selected frame indices: {indices}")

    return frames


# ──────────────────────────────────────────
# 2. 원통형 투영
# ──────────────────────────────────────────
def cylindrical_warp(img, f):
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    y_i, x_i = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    theta = (x_i - cx) / f

    x_src = f * np.tan(theta) + cx
    y_src = (y_i - cy) / np.cos(theta) + cy

    x_src = x_src.astype(np.float32)
    y_src = y_src.astype(np.float32)

    warped = cv.remap(
        img,
        x_src,
        y_src,
        cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    mask = cv.remap(
        np.ones((h, w), np.uint8) * 255,
        x_src,
        y_src,
        cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=0
    )

    coords = cv.findNonZero(mask)

    if coords is not None:
        x, y, ww, hh = cv.boundingRect(coords)
        warped = warped[y:y + hh, x:x + ww]

    return warped


# ──────────────────────────────────────────
# 3. ORB 매칭 + RANSAC translation
# ──────────────────────────────────────────
def orb_match(img1, img2, ratio_thresh=0.75):
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    orb = cv.ORB_create(nfeatures=5000)

    kp1, desc1 = orb.detectAndCompute(gray1, None)
    kp2, desc2 = orb.detectAndCompute(gray2, None)

    if desc1 is None or desc2 is None:
        return None, None, 0

    matcher = cv.BFMatcher(cv.NORM_HAMMING)
    raw = matcher.knnMatch(desc1, desc2, k=2)

    good = []

    for pair in raw:
        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < ratio_thresh * n.distance:
            good.append(m)

    if len(good) < 8:
        return None, None, len(good)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    return pts1, pts2, len(good)


def ransac_translation(pts1, pts2, iters=1000, thresh=3.0):
    if pts1 is None or pts2 is None or len(pts1) < 4:
        return None, 0

    diffs = pts1 - pts2

    best_count = 0
    best_t = None

    for _ in range(iters):
        idx = np.random.randint(len(diffs))
        t = diffs[idx]

        errors = np.linalg.norm(diffs - t, axis=1)
        inliers = errors < thresh
        count = int(inliers.sum())

        if count > best_count:
            best_count = count
            best_t = np.median(diffs[inliers], axis=0)

    return best_t, best_count


# ──────────────────────────────────────────
# 4. 프레임 이동량 추정
# ──────────────────────────────────────────
def estimate_pair_offsets(images):
    offsets = []

    for i in range(len(images) - 1):
        pts1, pts2, n_matches = orb_match(images[i], images[i + 1])
        t, inliers = ransac_translation(pts1, pts2)

        if t is None:
            fallback = offsets[-1] if offsets else np.array([30.0, 0.0], dtype=np.float32)
            offsets.append(fallback)
            print(f"[WARN] pair {i}-{i + 1}: fallback offset={fallback}")
        else:
            offsets.append(t)
            print(f"[INFO] pair {i}-{i + 1}: matches={n_matches}, inliers={inliers}, offset={t}")

    return np.array(offsets, dtype=np.float32)


def smooth_1d(values, k=5):
    values = np.asarray(values, dtype=np.float32)

    if len(values) < k:
        return values

    pad = k // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=np.float32) / k

    return np.convolve(padded, kernel, mode="valid")


# ──────────────────────────────────────────
# 5. keyframe 선택
# ──────────────────────────────────────────
def select_keyframes(images, target_dx_ratio=0.78):
    pair_offsets = estimate_pair_offsets(images)

    dx = pair_offsets[:, 0]
    dy = pair_offsets[:, 1]

    cum_x = np.array([0.0] + list(np.cumsum(dx)), dtype=np.float32)
    cum_y = np.array([0.0] + list(np.cumsum(dy)), dtype=np.float32)

    # y축 흔들림만 아주 약하게 안정화
    cum_y_smooth = smooth_1d(cum_y, k=5)

    total_span = abs(cum_x[-1])
    frame_w = images[0].shape[1]
    target_dx = frame_w * target_dx_ratio

    if total_span < target_dx:
        selected_idx = list(range(len(images)))
    else:
        n_keys = max(int(total_span / target_dx) + 1, 4)
        target_positions = np.linspace(cum_x[0], cum_x[-1], n_keys)

        selected_idx = []

        for target in target_positions:
            idx = int(np.argmin(np.abs(cum_x - target)))

            if not selected_idx or idx != selected_idx[-1]:
                selected_idx.append(idx)

        if selected_idx[0] != 0:
            selected_idx.insert(0, 0)

        if selected_idx[-1] != len(images) - 1:
            selected_idx.append(len(images) - 1)

    selected_images = [images[i] for i in selected_idx]
    selected_offsets = [(cum_x[i], cum_y_smooth[i]) for i in selected_idx]

    print(f"[INFO] Selected keyframes: {selected_idx}")

    return selected_images, selected_offsets


# ──────────────────────────────────────────
# 6. 색상 보정
# ──────────────────────────────────────────
def color_harmonization(img, reference_mean):
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB).astype(np.float32)

    mean = lab.reshape(-1, 3).mean(axis=0)

    gain = reference_mean / (mean + 1e-6)
    gain = np.clip(gain, 0.92, 1.08)

    lab[:, :, 0] *= gain[0]
    lab[:, :, 1] = (lab[:, :, 1] - 128.0) * gain[1] + 128.0
    lab[:, :, 2] = (lab[:, :, 2] - 128.0) * gain[2] + 128.0

    lab = np.clip(lab, 0, 255).astype(np.uint8)

    return cv.cvtColor(lab, cv.COLOR_LAB2BGR)


# ──────────────────────────────────────────
# 7. ghost 최소화 seam-cut blending
# ──────────────────────────────────────────
def paste_with_seam_blend(canvas, valid_mask, img, px, py, direction):
    h, w = img.shape[:2]
    canvas_h, canvas_w = valid_mask.shape

    sx0 = max(0, -px)
    sy0 = max(0, -py)
    sx1 = min(w, canvas_w - px)
    sy1 = min(h, canvas_h - py)

    dx0 = max(0, px)
    dy0 = max(0, py)
    dx1 = dx0 + (sx1 - sx0)
    dy1 = dy0 + (sy1 - sy0)

    if sx1 <= sx0 or sy1 <= sy0:
        return canvas, valid_mask

    src = img[sy0:sy1, sx0:sx1].astype(np.float32)
    src_mask = (src.sum(axis=2) > 0)

    dst = canvas[dy0:dy1, dx0:dx1].astype(np.float32)
    dst_mask = valid_mask[dy0:dy1, dx0:dx1]

    non_overlap = src_mask & (~dst_mask)
    overlap = src_mask & dst_mask

    # 새 영역은 그대로 복사
    dst[non_overlap] = src[non_overlap]

    if np.any(overlap):
        roi_h, roi_w = src_mask.shape

        # 경계에서만 아주 좁게 blending
        blend_w = min(max(roi_w // 45, 3), 8)

        x = np.arange(roi_w, dtype=np.float32)

        if direction >= 0:
            # 카메라가 오른쪽으로 이동하는 경우: 새 이미지의 왼쪽 경계만 blending
            alpha_line = np.clip(x / blend_w, 0.0, 1.0)
        else:
            # 반대 방향
            alpha_line = np.clip((roi_w - 1 - x) / blend_w, 0.0, 1.0)

        alpha = np.tile(alpha_line[None, :], (roi_h, 1))
        #alpha = cv.GaussianBlur(alpha, (5, 5), 0)
        alpha = alpha[..., None]

        blended = dst * (1.0 - alpha) + src * alpha

        # overlap에서는 넓게 평균내지 않고 seam 기준으로 거의 한쪽 선택
        dst[overlap] = blended[overlap]

    canvas[dy0:dy1, dx0:dx1] = np.clip(dst, 0, 255).astype(np.uint8)
    valid_mask[dy0:dy1, dx0:dx1] |= src_mask

    return canvas, valid_mask


# ──────────────────────────────────────────
# 8. Translation 기반 stitching
# ──────────────────────────────────────────
def stitch_translation_panorama(frames, focal=None):
    if len(frames) < 2:
        print("[ERROR] Need at least 2 frames.")
        sys.exit(1)

    h0, w0 = frames[0].shape[:2]
    f = focal or max(h0, w0) * 0.9

    print(f"[INFO] Focal length: {f}")

    warped_images = [cylindrical_warp(frame, f) for frame in frames]

    keyframes, offsets = select_keyframes(warped_images)

    ref_img = keyframes[len(keyframes) // 2]
    ref_lab = cv.cvtColor(ref_img, cv.COLOR_BGR2LAB).astype(np.float32)
    reference_mean = ref_lab.reshape(-1, 3).mean(axis=0)

    keyframes = [
        color_harmonization(img, reference_mean)
        for img in keyframes
    ]

    xs = np.array([p[0] for p in offsets], dtype=np.float32)
    ys = np.array([p[1] for p in offsets], dtype=np.float32)

    min_x = int(np.floor(xs.min()))
    max_x = int(np.ceil(xs.max()))
    min_y = int(np.floor(ys.min()))
    max_y = int(np.ceil(ys.max()))

    max_h = max(img.shape[0] for img in keyframes)
    max_w = max(img.shape[1] for img in keyframes)

    canvas_w = (max_x - min_x) + max_w
    canvas_h = (max_y - min_y) + max_h

    shift_x = -min_x
    shift_y = -min_y

    direction = 1 if xs[-1] >= xs[0] else -1

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    valid_mask = np.zeros((canvas_h, canvas_w), dtype=bool)

    # x 위치 순서대로 붙여서 덮어쓰기 방향을 안정화
    order = np.argsort(xs)

    for idx in order:
        img = keyframes[idx]
        ox, oy = offsets[idx]

        px = int(round(ox + shift_x))
        py = int(round(oy + shift_y))

        canvas, valid_mask = paste_with_seam_blend(
            canvas,
            valid_mask,
            img,
            px,
            py,
            direction
        )

    return canvas


# ──────────────────────────────────────────
# 9. crop
# ──────────────────────────────────────────
def auto_crop(img, fill_threshold=0.85):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    cols = (gray > 0).sum(axis=0) / img.shape[0]
    rows = (gray > 0).sum(axis=1) / img.shape[1]

    valid_cols = np.where(cols > fill_threshold)[0]
    valid_rows = np.where(rows > fill_threshold)[0]

    if len(valid_cols) == 0 or len(valid_rows) == 0:
        coords = cv.findNonZero((gray > 0).astype(np.uint8))

        if coords is None:
            return img

        x, y, w, h = cv.boundingRect(coords)

        return img[y:y + h, x:x + w]

    x0, x1 = valid_cols[0], valid_cols[-1]
    y0, y1 = valid_rows[0], valid_rows[-1]

    return img[y0:y1 + 1, x0:x1 + 1]


# ──────────────────────────────────────────
# 10. main
# ──────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--start", type=float, default=None)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--rotate", type=int, default=0)
    parser.add_argument("--focal", type=float, default=None)

    args = parser.parse_args()

    v3_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(v3_dir)

    if args.video is None:
        video_path = os.path.join(project_dir, "input_video", "input.mp4")
    elif os.path.isabs(args.video):
        video_path = args.video
    else:
        video_path = os.path.join(project_dir, args.video)

    output_path = os.path.join(v3_dir, "panorama_improved_v3.jpg")

    frames = extract_frames(
        video_path,
        n_frames=args.frames,
        start_sec=args.start,
        end_sec=args.end,
        rotate=args.rotate
    )

    if args.scale != 1.0:
        frames = [
            cv.resize(frame, None, fx=args.scale, fy=args.scale)
            for frame in frames
        ]

    result = stitch_translation_panorama(frames, focal=args.focal)
    result = auto_crop(result)

    cv.imwrite(output_path, result, [cv.IMWRITE_JPEG_QUALITY, 95])

    print(f"[INFO] Done! Saved as {output_path}")


if __name__ == "__main__":
    main()
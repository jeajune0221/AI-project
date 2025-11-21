import cv2
import time
import numpy as np
import mediapipe as mp
import os

# 설정하는 부분
USE_MIRROR = True
CAM_W, CAM_H = 1280, 720
PRINT_EVERY = 30 

# Mediapipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,     
    max_num_hands=2,             
    model_complexity=0,          
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

def ensure_folders():
    os.makedirs("dataset/left", exist_ok=True)
    os.makedirs("dataset/right", exist_ok=True)
    os.makedirs("dataset/other", exist_ok=True)

def detect_hand_roi(frame_bgr, roi_size=224, margin_ratio=0.15):
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    if not res.multi_hand_landmarks:
        return None, None, None

    lm = res.multi_hand_landmarks[0].landmark
    xs = (np.array([p.x for p in lm]) * w).astype(int)
    ys = (np.array([p.y for p in lm]) * h).astype(int)

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    bw, bh = x1 - x0, y1 - y0
    side = int(max(bw, bh) * (1 + margin_ratio))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2

    sx0 = max(0, cx - side // 2)
    sy0 = max(0, cy - side // 2)
    sx1 = min(w, sx0 + side)
    sy1 = min(h, sy0 + side)

    roi = frame_bgr[sy0:sy1, sx0:sx1]
    if roi.size == 0:
        return None, None, None

    roi224 = cv2.resize(roi, (roi_size, roi_size), interpolation=cv2.INTER_AREA)
    bbox = (sx0, sy0, sx1, sy1)
    lm_px = np.stack([xs, ys], axis=1)  
    return roi224, bbox, lm_px


def hand_mask_from_landmarks(frame_shape, lm_px):
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if lm_px is None:
        return mask
    hull = cv2.convexHull(lm_px.astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    return mask





def main():


    ensure_folders()
    counters = {"left": 0, "right": 0, "other": 0}
    files = os.listdir("dataset/left")
    leftn = len(files)
    counters["left"] = leftn
    files = os.listdir("dataset/right")
    rightn = len(files)
    counters["right"] = rightn
    files = os.listdir("dataset/other")
    othern = len(files)
    counters["other"] = othern

    current_label = ""

    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    cv2.namedWindow("cam", cv2.WINDOW_NORMAL)
    cv2.namedWindow("hand_only", cv2.WINDOW_NORMAL)
    cv2.namedWindow("debug", cv2.WINDOW_NORMAL)
    cv2.namedWindow("roi224", cv2.WINDOW_NORMAL)

    frame_idx = 0
    prev = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if USE_MIRROR:
            frame = cv2.flip(frame, 1)

        roi224, bbox, lm_px = detect_hand_roi(frame, roi_size=224, margin_ratio=0.15)

        # hand-only
        mask = hand_mask_from_landmarks(frame.shape, lm_px)
        hand_only = cv2.bitwise_and(frame, frame, mask=mask)

        # 디버그
        dbg = frame.copy()
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            cv2.rectangle(dbg, (x0, y0), (x1, y1), (0, 255, 0), 2)
        if lm_px is not None:
            for (x, y) in lm_px:
                cv2.circle(dbg, (int(x), int(y)), 2, (0, 0, 255), -1)

        # 화면 출력
        cv2.imshow("cam", frame)
        cv2.imshow("hand_only", hand_only)
        cv2.imshow("debug", dbg)
        if roi224 is not None:
            cv2.imshow("roi224", roi224)        
        #자동 저장
        
        if roi224 is not None and current_label != "":
            if frame_idx % 3 == 0:   # 3프레임마다 저장
                folder = f"dataset/{current_label}/"
                count = counters[current_label]
                filename = f"{current_label}_{count:05d}.png"
                save_path = folder + filename

                cv2.imwrite(save_path, roi224)
                counters[current_label] += 1
                print("Saved:", save_path)


        # FPS
        frame_idx += 1
        now = time.perf_counter()
        fps = 1.0 / (now - prev)
        prev = now

        if frame_idx % PRINT_EVERY == 0:
            print(f"FPS: {fps:.1f}")
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            current_label = "left"
            print("Label → LEFT")
        elif key == ord('2'):
            current_label = "right"
            print("Label → RIGHT")
        elif key == ord('3'):
            current_label = "other"
            print("Label → OTHER")
        elif key == ord('q'):
            break

    cap.release()
    hands.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# [전체 요약]
# - HandGestureCNN 로 best_model.pth 가중치 로드
# - Mediapipe로 손 랜드마크 검출
# - 손 박스(bbox) 기준으로 224x224 ROI 추출 후 CNN에 넣어
#   LEFT / RIGHT / OTHER + 확률(%)를 실시간으로 표시
# - 제스처(LEFT/RIGHT)가 '변하는 순간' + 쿨다운이 지났을 때만
#   pyautogui 로 키 입력(왼/오른 화살표) 전송

import os
import time
import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import torchvision.transforms as T
import pyautogui


# =========================================================
# 1. 디바이스 선택 (CUDA / MPS / CPU)
# =========================================================
def get_device():
    if torch.cuda.is_available():
        print("[Device] Using CUDA")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("[Device] Using Apple MPS")
        return torch.device("mps")
    else:
        print("[Device] Using CPU")
        return torch.device("cpu")


# =========================================================
# 2. CNN 모델 정의 (학습 때와 완전히 동일해야 함)
# =========================================================
class HandGestureCNN(nn.Module):
    """left / right / other 3클래스 분류용 CNN"""
    def __init__(self, num_classes: int = 3):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 3x224x224 -> 16x112x112
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 16x112x112 -> 32x56x56
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 32x56x56 -> 64x28x28
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        conv_out_dim = 64 * 28 * 28  # 50176

        self.classifier = nn.Sequential(
            nn.Linear(conv_out_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)              # (N, 64, 28, 28)
        x = x.view(x.size(0), -1)         # (N, 64*28*28)
        x = self.classifier(x)            # (N, 3)
        return x


# =========================================================
# 3. Mediapipe 손 검출
# =========================================================

# 화면은 거울처럼 보고 싶으셔서 USE_MIRROR=True 유지
USE_MIRROR = True

CAM_W, CAM_H = 1280, 720

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def detect_hand_bbox(frame_bgr, margin_ratio=0.2):
    """
    BGR 프레임에서 첫 번째 손의 박스 + 랜드마크 픽셀좌표를 계산
    반환:
      bbox: (x0, y0, x1, y1) 또는 None
      lm_px: (21,2) 랜드마크 (x,y) 배열 또는 None
    """
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    if not res.multi_hand_landmarks:
        return None, None

    lm = res.multi_hand_landmarks[0].landmark
    xs = (np.array([p.x for p in lm]) * w).astype(int)
    ys = (np.array([p.y for p in lm]) * h).astype(int)

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    bw, bh = x1 - x0, y1 - y0

    # 정사각형 박스 + 여유 margin
    side = int(max(bw, bh) * (1 + margin_ratio))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2

    sx0 = max(0, cx - side // 2)
    sy0 = max(0, cy - side // 2)
    sx1 = min(w, sx0 + side)
    sy1 = min(h, sy0 + side)

    bbox = (sx0, sy0, sx1, sy1)
    lm_px = np.stack([xs, ys], axis=1)

    return bbox, lm_px


# =========================================================
# 4. 메인: 카메라 열고 실시간 추론
# =========================================================
def main():
    device = get_device()

    # ---- 4-1. 모델 + 가중치 로드 ----
    model = HandGestureCNN(num_classes=3).to(device)

    weight_path = "best_model.pth"
    if not os.path.exists(weight_path):
        print(f"[에러] 모델 가중치 파일을 찾을 수 없습니다: {weight_path}")
        print("train_cnn.py 로 학습을 먼저 진행하여 best_model.pth 를 생성해 주세요.")
        return

    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"[모델] {weight_path} 로드 완료")

    # ---- 4-2. 전처리 (학습 때와 동일) ----
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]),
    ])

    # 0 → RIGHT, 1 → LEFT, 2 → OTHER  (사용자 기준으로 바꿔둔 상태)
    idx_to_label = {1: "LEFT", 0: "RIGHT", 2: "OTHER"}

    # ---- 4-3. 카메라 열기 ----
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # 맥 기준
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    if not cap.isOpened():
        print("[에러] 카메라를 열 수 없습니다.")
        return

    cv2.namedWindow("cam", cv2.WINDOW_NORMAL)
    cv2.namedWindow("roi224", cv2.WINDOW_NORMAL)

    print("실시간 추론 시작! (q 를 누르면 종료)")

    prev = time.perf_counter()
    frame_idx = 0

    # 예측 값 부드럽게 하기 위한 EMA
    ema_prob = None
    ema_alpha = 0.4  # 0.3~0.5 사이에서 튜닝 가능

    # 제스처 트리거/쿨다운 상태
    last_action_time = 0.0          # 마지막으로 키를 보낸 시각
    action_cooldown = 1.0           # 최소 간격(초)
    last_trigger_gesture = None     # 마지막으로 키를 보낸 제스처 (display_idx 기준 0/1)

    # 액션을 만들 최소 신뢰도 (EMA 기준) - 키보드 이벤트용
    min_action_conf = 0.5

    # 화면에 라벨을 LEFT/RIGHT로 보여줄 최소 신뢰도 (표시용)
    display_min_conf = 0.5

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[에러] 프레임을 읽을 수 없습니다.")
            break

        # 기본 OTHER
        pred_idx = 2
        display_idx = 2

        # 화면 표시용 프레임 (사람이 보는 것은 거울 모드)
        if USE_MIRROR:
            display_frame = cv2.flip(frame, 1)
        else:
            display_frame = frame.copy()

        # ---- 4-4. 손 박스 + 랜드마크 검출 (항상 원본 frame 기준) ----
        bbox, lm_px = detect_hand_bbox(frame, margin_ratio=0.2)

        label_text = "NO HAND"
        conf_text = ""

        if bbox is not None and lm_px is not None:
            x0, y0, x1, y1 = bbox
            roi = frame[y0:y1, x0:x1]
            if roi.size > 0:
                roi224 = cv2.resize(roi, (224, 224), interpolation=cv2.INTER_AREA)

                # CNN 입력용 전처리 (BGR -> RGB -> Tensor -> Normalize)
                roi_rgb = cv2.cvtColor(roi224, cv2.COLOR_BGR2RGB)
                inp = transform(roi_rgb)           # (3,224,224)
                inp = inp.unsqueeze(0).to(device)  # (1,3,224,224)

                with torch.no_grad():
                    logits = model(inp)
                    probs = torch.softmax(logits, dim=1)[0]  # (3,)

                probs_np = probs.cpu().numpy()
                if ema_prob is None:
                    ema_prob = probs_np
                else:
                    ema_prob = ema_alpha * probs_np + (1 - ema_alpha) * ema_prob

                pred_idx = int(np.argmax(ema_prob))
                pred_conf = float(ema_prob[pred_idx])

                # ----- 표시용 + 동작용 제스처 결정 -----
                # pred_conf가 낮으면 OTHER로 강제
                if pred_conf < display_min_conf:
                    display_idx = 2
                else:
                    display_idx = pred_idx

                label_text = idx_to_label[display_idx]
                conf_text = f"{pred_conf * 100:.1f}%"

                # ROI 창 (모델이 실제로 본 입력 확인용)
                cv2.imshow("roi224", roi224)

                # 디버그용 박스/랜드마크를 display_frame에 표시
                h, w = frame.shape[:2]
                if USE_MIRROR:
                    # x 좌표를 좌우 반전해서 그림
                    mx0 = w - x1
                    mx1 = w - x0
                    cv2.rectangle(display_frame, (mx0, y0), (mx1, y1), (0, 255, 0), 2)
                    for (x, y) in lm_px:
                        mx = w - x
                        cv2.circle(display_frame, (int(mx), int(y)), 2, (0, 0, 255), -1)
                else:
                    cv2.rectangle(display_frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                    for (x, y) in lm_px:
                        cv2.circle(display_frame, (int(x), int(y)), 2, (0, 0, 255), -1)

        # ---- 4-5. 텍스트 + FPS 표시 ----
        disp_text = label_text
        if conf_text:
            disp_text += f" ({conf_text})"

        cv2.putText(display_frame, disp_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        frame_idx += 1
        now = time.perf_counter()
        fps = 1.0 / (now - prev) if now > prev else 0.0
        prev = now

        cv2.putText(display_frame, f"FPS: {fps:.1f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("cam", display_frame)

        # ---- 4-6. 키보드 입력 처리 (제스처 디바운스 + 쿨다운) ----
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # HAND 없음 또는 OTHER → 트리거 리셋
        if display_idx == 2:
            last_trigger_gesture = None

        # LEFT 또는 RIGHT 인 경우만 키 입력 후보
        elif display_idx in (0, 1) and ema_prob is not None:
            # 확률이 너무 낮으면 무시
            if ema_prob[pred_idx] >= min_action_conf:
                # 같은 제스처로 이미 트리거된 상태면 다시 발사하지 않음
                if last_trigger_gesture != display_idx:
                    # 쿨다운 시간 지난 경우에만 키 전송
                    if now - last_action_time > action_cooldown:
                        if display_idx == 1:
                            # idx_to_label[1] == "LEFT"
                            pyautogui.press("left")
                            print("[ACTION] LEFT key sent")
                        elif display_idx == 0:
                            # idx_to_label[0] == "RIGHT"
                            pyautogui.press("right")
                            print("[ACTION] RIGHT key sent")

                        last_action_time = now
                        last_trigger_gesture = display_idx

    cap.release()
    hands.close()
    cv2.destroyAllWindows()
    print("종료합니다.")


if __name__ == "__main__":
    main()

import os
import cv2
import numpy as np

# --- 1. 이미지 경로 읽기 ---
folder = "dataset/left"
raw_paths = [
    os.path.join(folder, f)
    for f in os.listdir(folder)
    if f.lower().endswith(".png") or f.lower().endswith(".jpg")
]
raw_paths.sort()

# --- 2. 썸네일 생성 + 유효한 경로만 따로 저장 ---
valid_paths = []
thumbnails = []
for path in raw_paths:
    img = cv2.imread(path)
    if img is None:
        continue
    thumb = cv2.resize(img, (64, 64))
    thumbnails.append(thumb)
    valid_paths.append(path)

# 이제부터는 image_paths 대신 valid_paths를 사용
image_paths = valid_paths

# --- 3. 그리드 설정 ---
GRID_SIZE = 8         # 8x8 = 64개
THUMB = 64
PADDING = 2

canvas_h = GRID_SIZE * (THUMB + PADDING)
canvas_w = GRID_SIZE * (THUMB + PADDING)
canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

# --- 4. 페이지 ---
page = 0
per_page = GRID_SIZE * GRID_SIZE
total_pages = max(1, (len(thumbnails) + per_page - 1) // per_page)

while True:
    # --- 페이지 범위 보정 ---
    if page < 0:
        page = 0
    if page >= total_pages:
        page = total_pages - 1

    start = page * per_page

    # --- 캔버스 초기화 ---
    canvas[:] = 0

    # --- 5. 썸네일 그리드 구성 ---
    for idx in range(per_page):
        real_idx = start + idx
        if real_idx >= len(thumbnails):
            break

        img = thumbnails[real_idx]

        # 그리드 내 위치 계산
        row = idx // GRID_SIZE
        col = idx % GRID_SIZE

        y = row * (THUMB + PADDING)
        x = col * (THUMB + PADDING)

        # 썸네일 배치
        canvas[y:y+THUMB, x:x+THUMB] = img

        # 파일명 표시
        filename = os.path.basename(image_paths[real_idx])
        cv2.putText(canvas, filename, (x+2, y+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0,255,255), 1)

        # 인덱스도 함께 표시
        cv2.putText(canvas, f"#{real_idx}", (x+2, y+24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0,255,0), 1)

    # --- 6. 표시 ---
    cv2.imshow("check", canvas)
    print(f"Page {page+1}/{total_pages}")

    # --- 7. 키 입력 ---
    key = cv2.waitKey(0)

    if key == ord('q'):
        break
    elif key == ord('d'):   # 다음 페이지
        page += 1
    elif key == ord('a'):   # 이전 페이지
        page -= 1

cv2.destroyAllWindows()

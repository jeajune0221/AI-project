import os
import cv2
import numpy as np

# --- 1. 이미지 경로 읽기 ---
folder = "dataset/left"
image_paths = [
    os.path.join(folder, f)
    for f in os.listdir(folder)
    if f.lower().endswith(".png") or f.lower().endswith(".jpg")
]
image_paths.sort()

# --- 2. 썸네일 생성 ---
thumbnails = []
for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        continue
    thumb = cv2.resize(img, (64, 64))
    thumbnails.append(thumb)

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
    # 페이지 범위 보정
    if page < 0:
        page = 0
    if page >= total_pages:
        page = total_pages - 1

    start = page * per_page

    # 캔버스 초기화
    canvas[:] = 0

    # --- 5. 썸네일 그리드 구성 ---
    for idx in range(per_page):
        real_idx = start + idx
        if real_idx >= len(thumbnails):
            break

        img = thumbnails[real_idx]

        row = idx // GRID_SIZE
        col = idx % GRID_SIZE

        y = row * (THUMB + PADDING)
        x = col * (THUMB + PADDING)

        canvas[y:y+THUMB, x:x+THUMB] = img
        cv2.putText(canvas, str(real_idx), (x+2, y+10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,0), 1)


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

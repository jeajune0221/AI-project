import cv2
import time


cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION) # 카메라 여는 함수 여기서 0은 기반 카메라번호 CAP_AVFOUNDATION은 맥용 카메라 백엔드


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
frame_idx = 0
prev = time.perf_counter()
while True:
    ok, frame = cap.read() # ok는 프레임 읽기 성공 여부 frame은 실제 영상 데이터 
    if not ok: # ok false면 멈추기
        break
    frame = cv2.flip(frame, 1) # flip 1은 좌우 반전 0은 상하반전 -1은 상하 좌우 반전
    cv2.imshow("cam", frame) # cam이라는 창을 띄우고 frame 영상 띄우기
    frame_idx +=1
    
       
       
    now = time.perf_counter()
    fps = 1.0 / (now - prev)
    prev = now
    if(frame_idx%30== 0):
        print(f"FPS: {fps:.1f}")
    if cv2.waitKey(1) & 0xFF == ord('q'): # & 0xFF는 윈도우/맥 호환용 비트마스크입니다. q 키를 누르면 종료 waitKey(1)뭔말인줄 모르겠음
        break
cap.release() # C언어로 따지면 malloc
cv2.destroyAllWindows() #C언어로 따지면 return 0;

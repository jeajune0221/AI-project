import cv2


cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ok, frame = cap.read()
    if not ok:
        break


    frame = cv2.flip(frame, 1)


    cv2.imshow("cam", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

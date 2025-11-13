OpenCV로 카메라 입력을 받고, MediaPipe로 손의 21개 랜드마크를 검출한다.

손 영역을 Convex Hull로 감싸서 손만 흰색, 배경은 검정으로 마스크를 만든다.

이 마스크를 이용해 배경을 제거하고 손만 컬러로 남긴 hand_only 영상을 출력한다.

손 영역을 224×224 크기로 잘라 CNN 입력용 ROI(roi224)를 생성한다.

결과를 cam, hand_only, debug, roi224 창으로 실시간 확인하며 FPS를 측정한다.

import pyautogui
import time

time.sleep(3)  # 사용자에게 파워포인트 클릭할 시간 주기
pyautogui.press("right") # 다음 슬라이드로 넘기기
time.sleep(3)
pyautogui.press("left") # 이전 슬라이드로 넘기기

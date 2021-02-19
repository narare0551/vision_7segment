import cv2
import numpy as np
from matplotlib import pyplot as plt
import urllib
from urllib import request
import imutils
from collections import Counter
import datetime
import time

url = 'http://192.168.0.20:8080/shot.jpg'
global frame
global mask
global height
global width
global top_left_x
global top_left_y
global bottom_right_x
global bottom_right_y
global tH
global tW
global resized
templates = []
resized_mul = []
global minVal
global maxVal
global minLoc
global maxLoc
global zipped_template
found = []
global r
start_endXY = []
foundLoc = []
finalxy = []
methods = [ 'cv2.TM_CCOEFF_NORMED' , 'cv2.TM_CCOEFF']

def get_video(url):
    global frame
    global height
    global width
    global top_left_x
    global top_left_y
    global bottom_right_x
    global bottom_right_y
    imgResp = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
    # opencv에서 사용할 수 있는 형식으로 decode한다.

    frame = cv2.imdecode(imgNp, -1)

    height, width = frame.shape[:2]
    top_left_x = int(width / 4)
    top_left_y = int((height / 2) + (height / 6))
    bottom_right_x = int((width / 4) + (width / 5))
    bottom_right_y = int((height / 2) - (height / 55))
    return frame


def video_process(video):
    global mask
    global top_left_x
    global top_left_y
    global bottom_right_x
    global bottom_right_y
    cropped = video[bottom_right_y:top_left_y, top_left_x:bottom_right_x]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    # blur처리 한다
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # canny함수를 사용한다.
    canny = cv2.Canny(blur, 10, 70)
    # 이진화해서 나온 이미지 결과값을 mask에 저장한다.
    ret, mask = cv2.threshold(canny, 70, 255, cv2.THRESH_BINARY)


def show():
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)


def draw_rect():
    global frame
    cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), 255, 3)


def template():
    global templates
    global tH
    global tW
    global zipped_template

    for a in range(0, 10):
        # 저장경로에서 0-9까지의 template을 들고온다.
        templatepath = ('C:\\Users\\USER\\Desktop\\test\\verniersegment\\' + str(a) + 'vernier.png')  # trainImage
        # 이미지 템플렛을 로딩해서 참고한다.
        template = cv2.imread(templatepath, 1)  # trainImage
        # print('비교하는 숫자',a)

        # 이미지 template의 가로 세로를 구한다
        # grayscale로 변경하고
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        # 테두리를 감지한다  -> 인식률을 높여줌 (template이 이미 테두리 처리 된 거라서 일단 주석처리함)
        template = cv2.Canny(template, 50, 200)
        templates.append(template)
        numbers = range(0, 10)
        zipped_template = (templates, numbers)

        # 템플렛의 가로 세로를 구한다
        (tH, tW) = template.shape[:2]


def resize_video(scale):
    global mask
    global r

    global resized
    # mask처리된 부분을 점점점 작게  resize해서 resized에 저장한다.
    resized = imutils.resize(mask, width=int(mask.shape[1] * scale))
    # 비율 구하기 : 원래잘린 가로 길이 / 사이즈 바뀐 가로 길이
    r = float(resized.shape[1] / mask.shape[1])


def compare_match(m):
    global resized
    global minVal
    global maxVal
    global minLoc
    global maxLoc
    global found
    global r
    global start_endXY


    compare = cv2.matchTemplate(resized, template, m)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(compare)
    print('minVal, maxVal, minLoc, maxLoc', minVal, maxVal, minLoc, maxLoc)
    #if found == [] or maxVal > found[0]:
    if found == [] or maxVal > found[0]:
        found = (maxVal, maxLoc, r)
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
        # 시작 x,y를 묶는다
        start_endXY = [startX, startY, endX, endY]
    print('found', found)
    print("start_endXY", start_endXY)


def mark_found():
    global frame
    global start_endXY
    cv2.putText(mask, str('temp'), (start_endXY[0] + 10, start_endXY[1] + 10), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)
    cv2.rectangle(mask, (start_endXY[0], start_endXY[1]), (start_endXY[2], start_endXY[3]), (255, 0, 0), 2)


def count():


template()
while True:
    get_video(url)
    global frame
    global mask
    video_process(frame)
    draw_rect()

    for scale in np.linspace(0.5, 1.0, num=6)[::-1]:
        resize_video(scale)

        for template in templates:
            for meth in methods:
                m=eval(meth)
                compare_match(m)

            mark_found()
            show()
    print(len(templates))
    if cv2.waitKey(1) == 13:
        break
cv2.destroyAllWindows()

import cv2
import pymysql
from datetime import datetime
from pathlib import Path
from win11toast import toast

# connection 정보
conn = pymysql.connect(host = 'localhost', user = 'root', password = 'mysql', db = 'hci', charset = 'utf8')
curs = conn.cursor(pymysql.cursors.DictCursor)


# 모델 정보
BODY_PARTS = {"Neck": 1, "RShoulder": 2, "LShoulder": 5}
BASE_DIR = Path(__file__).resolve().parent
protoFile = str(BASE_DIR) + "/source/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = str(BASE_DIR) + "/source/pose_iter_160000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# 카메라 정보
capture = cv2.VideoCapture(0)
inputWidth = 320;
inputHeight = 240;
inputScale = 1.0 / 255;


#시간
#9:10,10:10,11:10..에 촬영
resv_time=[(9,10),(10,10),(11,10),(12,10),(13,10),(14,10),(15,10),(16,10),(17,10),(18,10)]
visit = [0,0,0,0,0,0,0,0,0,0]
ind = 0
curr_hour, curr_min = 0,0
prev_hour , prev_min =0,0
res_hour , res_min = 0,0

# 함수
def extractNRL(frame, points):
    # 모델로 15 점 추출
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    inpBlob = cv2.dnn.blobFromImage(frame, inputScale, (inputWidth, inputHeight), (0, 0, 0), swapRB=False, crop=False)
    imgb = cv2.dnn.imagesFromBlob(inpBlob)
    net.setInput(inpBlob)
    output = net.forward()

    # NRL 정보 추출
    mat = [1, 2, 5]
    for j in range(3):
        i = mat[j]
        probMap = output[0, i, :, :]
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        x = (frameWidth * point[0]) / output.shape[3]
        y = (frameHeight * point[1]) / output.shape[2]

        # NRL 검출한 결과가 0.1보다 크면 points 추가, 아니면 None
        if prob > 0.1:
            points.append((int(x), int(y)))
        else:
            points.append(None)
    pass


def isTriangle(points):
    if points[0] and points[1] and points[2]: # 3점 다 찍힘
        if (points[1][0] < points[0][0]) and (points[0][0]<points[2][0]): # x좌표가 RNL 순
            if points[0][1] < min(points[1][1],points[2][1]): # y 좌표가 N이 작음
                return True
    return False


def isTime():
    global ind
    curr_hour = datetime.today().hour
    curr_min = datetime.today().minute

    res_hour = resv_time[ind][0]
    res_min = resv_time[ind][1]
    global prev_hour,prev_min

    if (prev_hour == res_hour) and (prev_min == res_min):
        if (curr_hour!=res_hour) or (curr_min != res_min):
            ind+=1
    prev_hour, prev_min = curr_hour, curr_min

    if (curr_hour == res_hour) and (curr_min == res_min):
        if not visit[ind]:
            visit[ind]=1
            return True
    return False

def score_turtle(frame,curs):
    points = []
    # NRL 추출
    extractNRL(frame, points)
    res = frame
    # points가 측정 가능한 삼각형을 이룸
    if isTriangle(points):
        score=0
        x = abs(points[1][0] - points[2][0]) # 밑변
        h = abs(points[0][1] - (points[1][1] + points[2][1]) // 2) # 높이
        r = x // h # 비율
        if r <= 3:
            cv2.line(frame, points[0], points[1], (255, 0, 0), 2)
            cv2.line(frame, points[0], points[2], (255, 0, 0), 2)
            score = 100
        elif r <= 4:
            cv2.line(frame, points[0], points[1], (0, 255, 0), 2)
            cv2.line(frame, points[0], points[2], (0, 255, 0), 2)
            score = 80
        elif r <= 5:
            cv2.line(frame, points[0], points[1], (0, 255, 255), 2)
            cv2.line(frame, points[0], points[2], (0, 2555, 255), 2)
            score = 60
        elif r <= 6:
            cv2.line(frame, points[0], points[1], (0, 165, 255), 2)
            cv2.line(frame, points[0], points[2], (0, 165, 255), 2)

            # 알람 메시지 설정
            title = "안 좋은 자세 알람"
            message = "자세가 현재 굉장히 안 좋아요! 자세를 바로 세우세요!"

            # 알람 띄우기
            toast(title, message)

            score = 40
        else:
            cv2.line(frame, points[0], points[1], (0, 0, 255), 2)
            cv2.line(frame, points[0], points[2], (0, 0, 255), 2)
            #res = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)

            # 알람 메시지 설정
            title = "안 좋은 자세 알람"
            message = "자세가 현재 굉장히 안 좋아요! 자세를 바로 세우세요!"

            # 알람 띄우기
            toast(title, message)
            score = 20

        if isTime():
            # DB
            sql = """ insert into score(score, createdAt) values (%s, %s) """
            curs.execute(sql, (score, datetime.now()))
            conn.commit()
            pass

        pass

    return res


# 카메라 재생 , 아무 키나 누르면 끝난다.
while cv2.waitKey(1) < 0:

    hasFrame, frame = capture.read()
    if not hasFrame:
        cv2.waitKey()
        break

    res = score_turtle(frame, curs) #pose 점수 계산 함수 호출

    cv2.imshow("posture", res)
    pass

# 카메라 종료
capture.release()
cv2.destroyAllWindows()


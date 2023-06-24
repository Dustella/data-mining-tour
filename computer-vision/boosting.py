import cv2
import time
from prototype.proto import CompressiveTracker
# 读取视频文件
cap = cv2.VideoCapture('video.mp4')

# get fps of video
fps = cap.get(cv2.CAP_PROP_FPS)
stay = int(1000 / fps) 


# Boosting Tracker: cv2.TrackerBoosting_create()
# MIL Tracker: cv2.TrackerMIL_create()
# TLD Tracker: cv2.TrackerTLD_create()
# MedianFlow Tracker: cv2.TrackerMedianFlow_create()
# MOSSE Tracker: cv2.TrackerMOSSE_create()
# CSRT Tracker: cv2.TrackerCSRT_create()
# 读取第一帧
ret, frame = cap.read()

# 在第一帧上选择要跟踪的目标区域
bbox = cv2.selectROI(frame, False)

print(bbox)
print(type(bbox))

# 创建 KCF 跟踪器
# tracker = cv2.legacy.TrackerCSRT_create()
tracker = CompressiveTracker()
tracker.init(frame, bbox)

while True:
    start_time = time.time()
    # 读取下一帧
    ret, frame = cap.read()
    if not ret:
        break

    # 更新跟踪器并获取目标位置
    ok, bbox = tracker.update(frame)

    # 绘制矩形框来标记目标位置
    if ok:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2)
        # put text
        cv2.putText(frame, "Tracking KCF", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
    else:
        # 跟踪失败
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # 显示结果
    cv2.imshow('Tracking', frame)
    end_time = time.time()
    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

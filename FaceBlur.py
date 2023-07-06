import cv2
from cvzone.FaceDetectionModule import FaceDetector
cam = cv2.VideoCapture(0)
cam.set(3, 1920)
cam.set(4, 1080)

detector = FaceDetector(minDetectionCon=0.75)

while True:
    success, img = cam.read()
    img, bboxes = detector.findFaces(img, draw=True)

    if bboxes:
        for i, bbox in enumerate(bboxes):
            x,y,w,h = bbox['bbox']
            # if x<0: x=0
            # if y<0: y=0
            imgCrop = img[y:y+h, x:x+w]
            # cv2.imshow(f'Image Cropped {i}', imgCrop)
            imgBlur = cv2.blur(imgCrop,(35, 35))
            img[y:y+h, x:x+w] = imgBlur

    cv2.imshow("Camera", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break 
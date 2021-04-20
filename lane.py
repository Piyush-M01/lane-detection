import cv2
import numpy as np

def edge(f):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
    _,thresh1=cv2.threshold(gray,170,184,cv2.THRESH_BINARY_INV+ cv2.THRESH_TRIANGLE)
    return thresh1

def roi(f):
    h,w=f.shape
    reg=np.array([[(0,h),(0,h//2+25),(w,h//2+25),(w,h)]],np.int32)
    mask=np.zeros_like(f)
    cv2.fillPoly(mask,reg,255)
    img=cv2.bitwise_and(f,mask)
    return img



def houghlin(f,c):
    ope=cv2.morphologyEx(c,cv2.MORPH_OPEN,(5,5))
    dil=cv2.dilate(ope,(5,5))
    er=cv2.erode(dil,(5,5))
    lines = cv2.HoughLinesP(er,2,np.pi/180,40,minLineLength=50,maxLineGap=10)
    if lines is not None:
        for i in range(0,lines.shape[0]):
            for x1,y1,x2,y2 in lines[i]:
                cv2.line(f,(x1,y1),(x2,y2),(0,255,0),5)
    return f




vdo=cv2.VideoCapture("lane_vgt.mp4")
#out=cv2.VideoWriter("LANE_new.avi",cv2.VideoWriter_fourcc(*'XVID'),60.0,(640,480))
while True:
    _,frame=vdo.read()
    canimg=edge(frame)
    crop=roi(canimg)
    fr=houghlin(frame,crop)
    cv2.imshow("",fr)
    #out.write(fr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vdo.release()
cv2.destroyAllWindows()


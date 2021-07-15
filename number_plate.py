import cv2
import numpy as np
import pandas

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
state ={"DL":"delhi","ch":"chennai","ap":"andhara pradesh"}
img = cv2.imread("image.webp")
img_gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plate = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
number_plate = plate.detectMultiScale(img_gray,1.1,4)
print(number_plate)

for (x,y,w,h) in number_plate:
    a,b = (int(0.02*img.shape[0])), int(0.025*img.shape[1])
    plate = img[y+a:y+h-a,x+b:x+w-b,:]
     ## image processing
    kernel= np.ones((1,1),np.uint8)
    plate = cv2.dilate(plate,kernel,iterations=1)
    plate = cv2.erode(plate, kernel, iterations=1)
    cv2.imshow("number plate",plate)
    read = pytesseract.image_to_string(plate,lang='eng')
    read = ''.join(e for e in read if e.isalnum())
    s = read[0:2]
    try:
        print("car belongs to ",state[s] )
    except:
        print("not found state!")
    print(read)

    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(img, (x, y-40), (x + w, y), (255, 255, 255), -1)
    cv2.putText(img, read, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

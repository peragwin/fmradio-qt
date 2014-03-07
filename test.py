import numpy as np
import cv2

a = 0

while(1):
    a += .001
    img = np.ones((200,200))*a
    if a >= 1:
        a = 0
    cv2.imshow('image',img)
    if cv2.waitKey(1) & 0xFF == 27:
        cv2.destroyAllWindows()
        break
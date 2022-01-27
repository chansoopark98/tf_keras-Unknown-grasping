import glob
from imageio import imread
import cv2

# img = imread('utils/custom_data/pcd0100r.png')
img =  cv2.imread('utils/custom_data/jacquard_sample.png')

# img = cv2.line(img, (470, 253), (320, 320), (100, 100, 100), 2)


with open('utils/custom_data/jacquard_sample.txt') as f:
    for l in f:
        
        x, y, theta, w, h = [float(v) for v in l[:-1].split(';')]
        img = cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 0), 2)        
        
cv2.imshow('test', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 470.14815;253.89551;68.2615;25.5;25.6211
# 470.14815;253.89551;68.2615;25.5;38.4316

# 253 319.7 
# 309 324 
# 307 350 
# 251 345.7
# 255 324.877 
# 308 332 
# 313 295 
# 260 287.877
# 260 303.092 
# 311 309 
# 314 283 
# 263 277.092
# 258 279.048 
# 320 282 
# 321 261 
# 259 258.048
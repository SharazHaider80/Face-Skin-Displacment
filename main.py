import cv2
import mediapipe

drawingModule = mediapipe.solutions.drawing_utils
faceModule = mediapipe.solutions.face_mesh

circleDrawingSpec = drawingModule.DrawingSpec(thickness=2, circle_radius=1, color=(0, 255, 0))
lineDrawingSpec = drawingModule.DrawingSpec(thickness=2, color=(0, 255, 0))


X1 = []
Y1 = []
Z1 = []
C1 = []
with faceModule.FaceMesh(static_image_mode=True) as face:
    image = cv2.imread("Images/pic1.jpg")
    print(image.shape)
    results = face.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks != None:
        for faceLandmarks in results.multi_face_landmarks:
            drawingModule.draw_landmarks(image, faceLandmarks, faceModule.FACE_CONNECTIONS, circleDrawingSpec,
                                         lineDrawingSpec)
            for data1 in faceLandmarks.landmark:
                X1.append(round(data1.x,5))
                Y1.append(round(data1.y,5))
                Z1.append(round(-data1.z,5))
                C1.append(round(data1.x,5)**round(data1.y,5))

    cv2.imshow('Test image', image)



X = []
Y = []
Z = []
C = []
with faceModule.FaceMesh(static_image_mode=True) as face:
    image2 = cv2.imread("Images/pic2.jpg")
    print(image2.shape)
    results2 = face.process(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))

    if results2.multi_face_landmarks != None:
        for faceLandmarks in results2.multi_face_landmarks:
            drawingModule.draw_landmarks(image2, faceLandmarks, faceModule.FACE_CONNECTIONS, circleDrawingSpec,
                                         lineDrawingSpec)

            for data in faceLandmarks.landmark:
                X.append(round(data.x,5))
                Y.append(round(data.y,5))
                Z.append(round(-data.z,5))
                C.append(round(data.x,5)**round(data.y,5))



    cv2.imshow('Test image2', image2)

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Change the Size of Graph using
# Figsize
fig = plt.figure(figsize=(10,15))


ax = plt.axes(projection='3d')
# Creating array points using
# numpy


print('###################')
print(X)
# To create a scatter graph
#ax.scatter(X, Y, Z, c=C)
#ax.scatter(X1, Y1, Z1, c=C1)
lis = [0,0,0,0,0,0]
sh = 1
for x,y,z,u,v,w in zip(X,Y,Z,X1,Y1,Z1):
    #plt.plot([x,u],[y,v],[z,w], '-', color = 'y')
    #if sh == 0:
     #   plt.plot([(x + lis[0])/2, (u+ lis[3])/2], [(y+ lis[1])/2, (v+ lis[4])/2], [(z+ lis[2])/2, (w+ lis[5])/2], '<-', color='y')
    plt.plot([x+0.0150, u+0.0150], [y+0.0150, v+0.0150], [z+0.0150, w+0.0150], '<-', color='y')
    lis = [x,y,z,u,v,w]
    sh = 0
# trun off/on axis
plt.axis('on')

# show the graph
plt.show()


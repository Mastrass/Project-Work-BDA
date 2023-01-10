import cv2
import numpy as np

#video
#cap= cv2.VideoCapture('D:\Gabriele\Download\X2Download.app-BMPCC Cinematic 4K_ London.mp4')

#stream webcam
cap= cv2.VideoCapture(0)

#immagini
#path_img = r'C:\Users\Gabriele\Desktop\persone.jfif'
#foto = cv2.imread(path_img)


whT=320
#confThr1eshold=0.5
confThreshold=0.5
nmsThreshold=0.3

classesFile = 'coco.names'
classNames =[]

with open (classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
    print(classNames)
#   print(len(classNames))

modelConfiguration = 'yolov3-tiny.cfg'
modelWeights = 'yolov3-tiny.weights'

net = cv2.dnn.readNetFromDarknet(mod)  #rete neurale
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs,img):
    hT, wT, cT = img.shape
    bbox= []
    classIds= []
    confs = []

    for output in outputs:
        for det in output:
            scores= det[5:]
            classId = np.argmax(scores)
            confidence =scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]*wT), int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)
    num_obj = len(classIds)
    print("number of object: ", num_obj)

    for i in indices:
        box=bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                    (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)
        cv2.putText(img, f'object detected: {len(classIds)}',(20,40), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)



while True:
    success, img = cap.read()


    blob=cv2.dnn.blobFromImage(img,1/255,(whT,whT),[0,0,0],1,crop=False)  #preprocessing images and preparing to classification
    net.setInput(blob)



    layerNames=net.getLayerNames()
   # print(layerNames)
    outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]
#   print(outputNames)
#    print(net.getUnconnectedOutLayers())
    outputs=net.forward(outputNames)
#    print(outputs[0].shape)
#    print(outputs[1].shape)
#    print(outputs[2].shape)
#    print(outputs[0][0])

    findObjects(outputs,img)
    cv2.imshow('Image',img)
    cv2.waitKey(1)


    #findObjects(outputs,foto)
    #cv2.imshow('Image',foto)
    #cv2.waitKey(1)
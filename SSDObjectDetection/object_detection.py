# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:02:40 2019

@author: MaximusMinimus
"""

import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap  # classes encoding
from ssd import build_ssd 
import imageio # process imgs 

# Detect Function - frame by frame
# Args - image, ssd neural network, transform images for neural network compatibility
def detect(frame, net, transform):
    height, width = frame.shape[:2]  # get dims
    frame_t = transform(frame)[0] # apply transform for right dims - get first element of two - transform frame
    x = torch.from_numpy(frame_t).permute(2,0,1) # transform numpy frame to torch tensor - INVERSE THE COLOR ORDER FROM RBG INTO GRB
    x = Variable(x.unsqueeze(0)) # add fake dimension corresponding to the batch - nn accepts only batches of inputs - batch always first dimension
    #    ^^^^^ transform into torch variable
    
    y = net(x)   # feed x into neural network to get output
    detections = y.data  # create detections tensor [batch of outputs, number of classes, number of occurence, tuple(score, x0, x1, y0, y1)]
    scale = torch.Tensor([width, height, width, height]) # create w,h,w,h tensor - for normalization scaling (two corners of the rectangle)
    for i in range(detections.size(1)):
        j = 0
        while detections[0,i,j,0] >= 0.6:
            pt = (detections[0, i, j, 1:] * scale).numpy() # apply normalization 
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2)
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)    
            j += 1        
    return frame        

# Crate SSD Neural network
net = build_ssd('test')
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage)) # load weights 
   
# Creating the transformation
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))  # target siz o the imgs, scale colors

# Object Detection
reader = imageio.get_reader('funny_dog.mp4')
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('output.mp4', fps = fps)
for i, frame in enumerate(reader):
    frame = detect(frame, net.eval(), transform)
    writer.append_data(frame)
    print(i)
writer.close()    

















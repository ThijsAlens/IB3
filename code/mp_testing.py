import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
import time
import multiprocessing as mp
from wall_aproximation import mainfunc
import multiprocessing as mp
import os
import sys

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

x1, y1, x2, y2 = 0, 0, 320, 480 

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_vits14').to(DEVICE).eval()

transform = Compose([
    Resize(
        width=128,
        height=128,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method='lower_bound',
        image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])

cap = cv2.VideoCapture(0)
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

i = True
while i:
    curr_time = time.time()
    
    ret, raw_frame = cap.read()
    if not ret:
        break

    raw_frame1 = raw_frame[y1:y2, x1:x2]
    raw_frame2 = raw_frame[240+y1:240+y2, x1:x2]

    pipe1_r, pipe1_w = os.pipe()
    pipe2_r, pipe2_w = os.pipe()

    pid1 = os.fork()

    if (pid1 == 0): #in child 
        frame1 = cv2.cvtColor(raw_frame1, cv2.COLOR_BGR2RGB) / 255.0

        frame1 = transform({'image': frame1})['image']
        frame1 = torch.from_numpy(frame1).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            depth1 = depth_anything(frame1)

        depth1 = F.interpolate(depth1[None], (raw_frame1.shape[0], raw_frame1.shape[1]), mode='bilinear', align_corners=False)[0, 0]    
        depth1 = (depth1 - depth1.min()) / (depth1.max() - depth1.min()) * 255.0

        depth1 = depth1.cpu().numpy().astype(np.uint8)

        l = len(depth1)
        os.close(pipe1_w)
        os.write(pipe1_r, l)    #write the length

        os.write(pipe1_r, depth1)   #write the data

        sys.exit(0) #terminate child process
    else:   #in parent
        pid2 = os.fork()
        if (pid2 == 0):
            frame2 = cv2.cvtColor(raw_frame2, cv2.COLOR_BGR2RGB) / 255.0

            frame2 = transform({'image': frame2})['image']
            frame2 = torch.from_numpy(frame2).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                depth2 = depth_anything(frame2)

            depth2 = F.interpolate(depth2[None], (raw_frame2.shape[0], raw_frame2.shape[1]), mode='bilinear', align_corners=False)[0, 0]    
            depth2 = (depth2 - depth2.min()) / (depth2.max() - depth2.min()) * 255.0

            depth2 = depth2.cpu().numpy().astype(np.uint8)

            l = len(depth2)
            os.close(pipe2_w)
            os.write(pipe2_r, l)    #write the length

            os.write(pipe2_r, depth2)   #write the data

            sys.exit(0) #terminate child process

        else:
            os.close(pipe1_r)
            os.close(pipe2_r)

            l1 = os.read(pipe1_w, len(int))
            l2 = os.read(pipe2_w, len(int))

            depth1 = os.read(pipe1_w, l1)
            depth2 = os.read(pipe2_w, l2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                i = False

cap.release()
cv2.destroyAllWindows()
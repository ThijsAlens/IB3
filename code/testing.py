import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
import time
import multiprocessing as mp
#from wall_aproximation import mainfunc

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

def worker_function(parent_connection, raw_frame, transform, DEVICE, depth_anything):
    print("start process x")
    frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB) / 255.0
    print(1)
    frame = transform({'image': frame})['image']
    frame = torch.from_numpy(frame).unsqueeze(0).to(DEVICE)
    print(2)
    with torch.no_grad():
        print(7)
        depth = depth_anything(frame)
    print(3)
    #print("min: ", depth.min().item(), ", max: ", depth.max().item())

    depth = F.interpolate(depth[None], (raw_frame.shape[0], raw_frame.shape[1]), mode='bilinear', align_corners=False)[0, 0]    
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    print(4)
    depth = depth.cpu().numpy().astype(np.uint8)

    print("sending data")
    parent_connection.send(depth)
    parent_connection.close()

if __name__ == '__main__':

    x1, y1, x2, y2 = 0, 0, 320, 480 

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

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_vits14').to(DEVICE).eval()

    print("voor")
    cap = cv2.VideoCapture(0)
    print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("na")

    while True:
        processes = []
        connections = []

        curr_time = time.time()

        ret, raw_frame = cap.read()
        if not ret:
            break

        raw_frame1 = raw_frame[y1:y2, x1:x2]
        raw_frame2 = raw_frame[y1:y2, 320+x1:320+x2]

        print("creating process 1")
        connections.append(mp.Pipe())
        processes.append(mp.Process(target=worker_function, args=(connections[0][1], raw_frame1, transform, DEVICE, depth_anything)))
        processes[0].start()
        print("process 1 created")

        print("creating process 2")
        connections.append(mp.Pipe())
        processes.append(mp.Process(target=worker_function, args=(connections[1][1], raw_frame2, transform, DEVICE, depth_anything)))
        processes[1].start()
        print("process 2 created")

        for process in processes:
            process.join()

        results = np.empty((cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        results[y1:y2, x1:x2] = connections[0][0].recv()
        results[y1:y2, 320+x1:320+x2] = connections[1][0].recv()

        #mainfunc(results)

        depth_color = cv2.applyColorMap(results, cv2.COLORMAP_INFERNO)

        cv2.imshow('Depth Map', depth_color)

        print('FPS:', 1 / (time.time() - curr_time))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
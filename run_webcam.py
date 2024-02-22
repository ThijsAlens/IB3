import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
import time

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

#x1, y1, x2, y2 = 0, 140, 640, 340 

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

counter = 0

while True:
    curr_time = time.time()
    
    ret, raw_frame = cap.read()
    if not ret:
        break

    #raw_frame = raw_frame[y1:y2, x1:x2]
    frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB) / 255.0

    frame = transform({'image': frame})['image']
    frame = torch.from_numpy(frame).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        depth = depth_anything(frame)

    depth = F.interpolate(depth[None], (raw_frame.shape[0], raw_frame.shape[1]), mode='bilinear', align_corners=False)[0, 0]    
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

    depth = depth.cpu().numpy().astype(np.uint8)
    depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

    counter += 1
    if counter == 100:
        np.savetxt('snapshot_data', depth, fmt='%4.6f', delimiter=' ')
        
    cv2.imshow('Depth Map', depth_color)

    print('FPS:', 1 / (time.time() - curr_time))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
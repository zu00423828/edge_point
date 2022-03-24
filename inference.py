from model.kp_model import KPDetector
import numpy as np
import cv2
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def kp2gaussian(kp, img_size):
    kp = (kp+1)/2*(img_size-1)
    return kp


model = KPDetector().to(device)
model.state_dict(torch.load('model.pth'))
model.eval()
img = cv2.imread('frames/000016.jpg')
img_size = np.array(img.shape[:2])
model_input = (cv2.resize(img, (256, 256))/255).astype(np.float32)
with torch.no_grad():
    model_input = torch.tensor(model_input).unsqueeze(0).permute(0, 3, 1, 2)
    pred = model(model_input.to(device))
    currect_point = kp2gaussian(pred[0].cpu().numpy(), img_size)
    for p in currect_point:
        cv2.circle(img, (int(p[0]), int(p[1])), 3, (255, 0, 0), 3)
    cv2.imwrite('test.png', img)


import torch
from PIL import Image
from torchvision import datasets, transforms
import numpy as np
import argparse
import sys
import cv2

from vision.ssd.config.fd_config import define_img_size

sys.path.insert(0, 'D:/1Hung/01_DATN/Code/Emotion_Recognition/vision')
parser = argparse.ArgumentParser(
    description='detect_video')

parser.add_argument('--net_type', default="RFB", type=str,
                    help='The network architecture ,optional: RFB (higher precision) or slim (faster)')
parser.add_argument('--input_size', default=160, type=int,
                    help='define network input size,default optional value 128/160/320/480/640/1280')
parser.add_argument('--threshold', default=0.7, type=float,
                    help='score threshold')
parser.add_argument('--candidate_size', default=1000, type=int,
                    help='nms candidate size')
parser.add_argument('--path', default="imgs", type=str,
                    help='imgs dir')
parser.add_argument('--test_device', default="cpu", type=str,
                    help='cuda:0 or cpu')
parser.add_argument('--video_path', default="/home/linzai/Videos/video/16_1.MP4", type=str,
                    help='path of video')
args = parser.parse_args()

input_img_size = args.input_size
define_img_size(input_img_size)  # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'

from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
from vision.utils.misc import Timer

IMG_SIZE =  224 

test_transforms = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ]
)

idx_to_class={0: 'angry', 1: 'disgusted', 2: 'happy', 3: 'neutral', 4: 'sad'}


label_path = "./models/voc-model-labels.txt"
PATH='models/affectnet_emotions/enet_b0_5_best.pt'

net_type = args.net_type

# cap = cv2.VideoCapture(args.video_path)  # capture from video
cap = cv2.VideoCapture(cv2.CAP_DSHOW)  # capture from camera

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)
test_device = args.test_device

candidate_size = args.candidate_size
threshold = args.threshold

if net_type == 'slim':
    model_path = "models/pretrained/version-slim-320.pth"
    # model_path = "models/pretrained/version-slim-640.pth"
    net = create_mb_tiny_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_mb_tiny_fd_predictor(net, candidate_size=candidate_size, device=test_device)
elif net_type == 'RFB':
    model_path = "models/pretrained/version-RFB-320.pth"
    # model_path = "models/pretrained/version-RFB-640.pth"
    net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=candidate_size, device=test_device)
else:
    print("The net type is wrong!")
    sys.exit(1)
net.load(model_path)

model = torch.load(PATH, map_location=torch.device(test_device))

timer = Timer()
sum = 0
frame_count = 0
dict_emo = {
    'angry': 0, 'disgusted': 0, 'happy': 0, 'neutral': 0, 'sad': 0
}
while True:
    ret, orig_image = cap.read()
    if orig_image is None:
        print("end")
        break
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    timer.start()
    boxes, labels, probs = predictor.predict(image, candidate_size / 2, threshold)
    interval = timer.end()
    # print('Time: {:.6f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
    print(frame_count)
    
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label = f" {probs[i]:.2f}"
        x = int(box[0])
        y = int(box[1])
        w = int(box[2]) - int(box[0])
        h = int(box[3]) - int(box[1])
        aa = max(w, h)
        
        og_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        face = og_rgb[y-40:y+aa+20, x-40:x-40+aa]
        im = Image.fromarray(np.uint8(face))
        im = im.resize((224,224))
        
        scores = model(test_transforms(im).unsqueeze(0))
        scores=scores[0].data.cpu().numpy()
        state = idx_to_class[np.argmax(scores[:5])]
        dict_emo[state] += 1
        # cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[0])+aa, int(box[1])+aa), (0, 255, 0), 4)
        cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 4)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(orig_image,state,(x+10,y+15), font, 0.5, (255,255,255), 2, cv2.LINE_AA)
        
    orig_image = cv2.resize(orig_image, None, None, fx=0.8, fy=0.8)
    sum += boxes.size(0)
    # cv2.imshow('Cam 0', orig_image)
    frame_count += 1
    if frame_count == 300:
        max_emo = max(dict_emo, key= lambda x: dict_emo[x])
        print(f'max_emo: {max_emo}')
        frame_count = 0
        dict_emo = {'angry': 0, 'disgusted': 0, 'happy': 0, 'neutral': 0, 'sad': 0 }
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
# print("all face num:{}".format(sum))

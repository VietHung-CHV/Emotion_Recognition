
import torch
from PIL import Image
from torchvision import datasets, transforms
import numpy as np
import argparse
import sys
import cv2
import time

from vision.ssd.config.fd_config import define_img_size

parser = argparse.ArgumentParser(
    description='detect_video')

parser.add_argument('--net_type', default="RFB", type=str,
                    help='The network architecture ,optional: RFB (higher precision) or slim (faster)')
parser.add_argument('--input_size', default=320, type=int,
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
# train_transforms = transforms.Compose(
#     [
#         transforms.Resize((IMG_SIZE,IMG_SIZE)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#     ]
# )

# train_dir = 'datasets/custom_datasets/train'
# train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
# # class_to_idx=train_dataset.class_to_idx

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
prev_frame_time = 0

new_frame_time = 0
frame_t=0
list_emo = []
sum = 0
while True:
    ret, orig_image = cap.read()
    frame_t+=1
    if orig_image is None:
        print("end")
        break
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    timer.start()
    boxes, labels, probs = predictor.predict(image, candidate_size / 2, threshold)
    interval = timer.end()
    print('Time: {:.6f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
    
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label = f" {probs[i]:.2f}"
        x = int(box[0])
        y = int(box[1])
        w = int(box[2]) - int(box[0])
        h = int(box[3]) - int(box[1])
        aa = max(w, h)
        
        # x = int(box[0])
        # y = int(box[1])
        # w = int(box[2]) - int(box[0])
        # h = int(box[3]) - int(box[1])
        # aa = max(w, h)
        # face = orig_image[y:int(box[3]), x:int(box[2])]
        og_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        face = og_rgb[y-40:y+aa+20, x-40:x-40+aa]
        im = Image.fromarray(np.uint8(face))
        im = im.resize((224,224))
        # im.save('res/face.jpg')
        # face = cv2.resize(face, (48,48)) 
        # face = face/255.0
        # print(im.size)
        # print(test_transforms(im))
        
        scores = model(test_transforms(im).unsqueeze(0)) #run_time=0.25s
        
        
        class_prob = torch.softmax(scores, dim=1)
        # print(class_prob)
        # get most probable class and its probability:
        class_prob, topclass = torch.max(class_prob, dim=1)
        sco = class_prob.tolist()[0]
        # print(f'topclass: {topclass.numpy()[0]}')
        # get class names
        state = idx_to_class[topclass.numpy()[0]]
        list_emo.append(state)
        if len(list_emo)>10:
            list_emo.pop(0)
        set_list_emo = set(list_emo)
        cnts = 0
        for i in set_list_emo:
            if list_emo.count(i)>cnts:
                cnts = list_emo.count(i)
                best_emo = i
        
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = "FPS: "+str(fps)
        
        # proba = torch.softmax(scores, 0)
        # scores=scores[0].data.cpu().numpy()
        # print(scores)
        # state = idx_to_class[np.argmax(scores[:5])]
        
        if best_emo == "angry":
            best_emo = "tuc gian"
        elif best_emo =="disgusted":
            best_emo = "kho chiu"
        elif best_emo == "happy":
            best_emo = "vui ve"
        elif best_emo == "neutral":
            best_emo = "binh thuong"
        else:
            best_emo = "buon"
        
        best_emo = best_emo #+ " - " + "{:.3f}".format(sco)
        # cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[0])+aa, int(box[1])+aa), (0, 255, 0), 4)
        # cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 4)
        font = cv2.FONT_HERSHEY_SIMPLEX
        if frame_t %10==0 and frame_t >=10:
            cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 4)
            cv2.putText(orig_image,best_emo,(x+10,y+15), font, 0.5, (255,255,255), 2, cv2.LINE_AA)
            cur_emo = best_emo 
        elif frame_t %10!=0 and frame_t >10:
            cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 4)
            cv2.putText(orig_image,cur_emo,(x+10,y+15), font, 0.5, (255,255,255), 2, cv2.LINE_AA)
        
        # cv2.putText(orig_image, fps, (7, 70), font, 1, (100, 255, 0), 1, cv2.LINE_AA)
        # cv2.putText(orig_image, label,
        #             (box[0], box[1] - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             0.5,  # font scale
        #             (0, 0, 255),
        #             2)  # line type
    orig_image = cv2.resize(orig_image, None, None, fx=0.8, fy=0.8)
    sum += boxes.size(0)
    cv2.imshow('Cam 0', orig_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
print("all face num:{}".format(sum))

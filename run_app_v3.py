from pathlib import Path
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.icons import Emoji
import pygame
from PIL import ImageTk, Image, ImageOps
import glob
import time
from mutagen.mp3 import MP3
import torch
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


# global paused
# paused = False

class MediaPlayer(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(fill=BOTH, expand=YES)
        self.hdr_var = ttk.StringVar()
        self.elapsed_var = ttk.DoubleVar(value=0)
        self.remain_var = ttk.DoubleVar(value=190)
        pygame.init()
        self.paused = False
        self.stopped = False
        self.song_length=None
        # left panel
        self.left_panel = ttk.Frame(self, padding=(2, 1))
        self.left_panel.pack(side=LEFT, fill=BOTH, expand=YES)
        
        # right panel
        self.right_panel = ttk.Frame(self, style='bg.TFrame')
        self.right_panel.pack(side=RIGHT, fill=Y, expand=YES)
        
        self.tree = self.create_tree_widget()
        # self.create_header()
        self.create_media_window()
        self.create_progress_meter()
        self.create_buttonbox()

    def create_header(self):
        """The application header to display user messages"""
        self.hdr_var.set("Open a file to begin playback")
        lbl = ttk.Label(
            master=self, 
            textvariable=self.hdr_var, 
            bootstyle=(LIGHT, INVERSE),
            padding=10
        )
        lbl.pack(fill=X, expand=YES)
        # lbl.grid()

    def create_media_window(self):
        """Create frame to contain media"""
        self.media = ttk.Label(self)
        self.media.grid_propagate(0)
        self.media.pack(fill=BOTH, expand=YES)

    def create_progress_meter(self):
        """Create frame with progress meter with lables"""
        container = ttk.Frame(self)
        container.pack(fill=X, expand=YES, pady=10)

        self.elapse = ttk.Label(container, text='00:00')
        self.elapse.pack(side=LEFT, padx=10)

        self.scale = ttk.Scale(
            master=container, 
            command=self.slide, 
            bootstyle=SECONDARY,
            from_=0,
            to=100,
            orient=HORIZONTAL,
            value=0,
            length=360,
        )
        self.scale.pack(side=LEFT, fill=X, expand=YES)

        self.remain = ttk.Label(container, text='03:10')
        self.remain.pack(side=LEFT, fill=X, padx=10)

    def create_buttonbox(self):
        """Create buttonbox with media controls"""
        container = ttk.Frame(self)
        container.pack(fill=X, expand=YES)
        ttk.Style().configure('TButton', font="-size 14")

        rev_btn = ttk.Button(
            master=container,
            text=Emoji.get('black left-pointing double triangle with vertical bar'),
            padding=10,
            command=self.prev,
        )
        rev_btn.pack(side=LEFT, fill=X, expand=YES)

        play_btn = ttk.Button(
            master=container,
            text=Emoji.get('black right-pointing triangle'),
            padding=10,
            command=self.play,
        )
        play_btn.pack(side=LEFT, fill=X, expand=YES)

        fwd_btn = ttk.Button(
            master=container,
            text=Emoji.get('black right-pointing double triangle with vertical bar'),
            padding=10,
            command=self.next,
        )
        fwd_btn.pack(side=LEFT, fill=X, expand=YES)

        pause_btn = ttk.Button(
            master=container,
            text=Emoji.get('double vertical bar'),
            padding=10,
            command=self.pause,
        )
        pause_btn.pack(side=LEFT, fill=X, expand=YES)        

        stop_btn = ttk.Button(
            master=container,
            text=Emoji.get('black square for stop'),
            padding=10,
            command=self.stop,
        )
        stop_btn.pack(side=LEFT, fill=X, expand=YES)                      


    def on_progress(self, val: float):
        """Update progress labels when the scale is updated."""
        elapsed = self.elapsed_var.get()
        remaining = self.remain_var.get()
        total = int(elapsed + remaining)

        elapse = int(float(val) * total)
        elapse_min = elapse // 60
        elapse_sec = elapse % 60

        remain_tot = total - elapse
        remain_min = remain_tot // 60
        remain_sec = remain_tot % 60

        self.elapsed_var.set(elapse)
        self.remain_var.set(remain_tot)

        self.elapse.configure(text=f'{elapse_min:02d}:{elapse_sec:02d}')
        self.remain.configure(text=f'{remain_min:02d}:{remain_sec:02d}')

    def create_tree_widget(self):
        
        # treeview frame
        container = ttk.Frame(self.left_panel)
        container.pack(pady=20)
        # Treeview Scrollbar
        tree_scroll = ttk.Scrollbar(container)
        tree_scroll.pack(side=RIGHT,fill=Y)
        #treeview
        tv = ttk.Treeview(container, yscrollcommand=tree_scroll.set, show='headings', height=16)
        tv.configure(columns=(
            'name'
        ))
        tv.column('name', width=650, stretch=True)

        for col in tv['columns']:
            tv.heading(col, text=col.title(), anchor=W)

        tv.pack(fill=X, pady=1)
        tree_scroll.config(command=tv.yview)
         
        return tv
    
    def add_playlist(self, playlist_path):
        self.elapse.configure(text='00:00')
        self.remain.config(text="00:00")
        self.scale.config(value=0)
        self.playlist_path = playlist_path
            
        for ix, song in enumerate(glob.glob(self.playlist_path+"/*")):
            song = song.replace(self.playlist_path, "")
            song = song.replace(".mp3", "")
            self.tree.insert('',END,ix,values=(song)) 
        # self.tree.insert('', END, 12)
        self.tree.selection_set(0)
        self.play()
        
        return int(self.song_length)
    
    def remove_playlist(self):
        if len(self.tree.get_children())>0:
            for x in self.tree.get_children():
                self.tree.delete(x)
        
    
    def play(self):
        cur_item = self.tree.focus()
        song = self.tree.item(cur_item)
        if len(song['values']) == 0:
            song_id = self.tree.selection()[0]
            song = self.tree.item(song_id,"values")[0]
        else:
            song = song['values'][0]
        song = f'{self.playlist_path}/{song}.mp3'
        song_mut = MP3(song)
        
        self.song_length = song_mut.info.length
        self.paused = False
        pygame.mixer.music.load(song)
        pygame.mixer.music.play(loops=0)
        self.play_time()


    def stop(self):
        self.elapse.configure(text='00:00')
        self.remain.config(text="00:00")
        self.scale.config(value=0)
        self.stopped = True
        pygame.mixer.music.stop()
        
    
    def pause(self):
        if self.paused:
            pygame.mixer.music.unpause()
            self.paused = False
        else:
            pygame.mixer.music.pause()
            self.paused = True
        
    def next(self):
        self.elapse.configure(text='00:00')
        self.remain.config(text="00:00")
        self.scale.config(value=0)
        
        song_id = self.tree.selection()[0]
        next_song_id = int(song_id) + 1
        if next_song_id > len(self.tree.get_children()):
            next_song_id = 0
        song = self.tree.item(str(next_song_id),"values")[0]
        self.tree.selection_set(str(next_song_id))
        song = f'{self.playlist_path}/{song}.mp3'
        self.song_length = MP3(song).info.length
        pygame.mixer.music.load(song)
        pygame.mixer.music.play(loops=0)
        
    
    def prev(self):
        self.elapse.configure(text='00:00')
        self.remain.config(text="00:00")
        self.scale.config(value=0)
        
        song_id = self.tree.selection()[0]
        prev_song_id = int(song_id) - 1
        if prev_song_id < 0:
            prev_song_id = len(self.tree.get_children())-1
        song = self.tree.item(str(prev_song_id),"values")[0]
        self.tree.selection_set(str(prev_song_id))
        song = f'{self.playlist_path}/{song}.mp3'
        self.song_length = MP3(song).info.length
        pygame.mixer.music.load(song)
        pygame.mixer.music.play(loops=0)
        
    
    def select_all(self):
        for item in self.tree.get_children():
            name = self.tree.item(item,"values")[0]
            print(name)
        return 0
    
    def slide(self, val: float):
        song_id = self.tree.selection()[0]
        song = self.tree.item(str(song_id),"values")[0]
        song = f'{self.playlist_path}/{song}.mp3'
        
        pygame.mixer.music.load(song)
        pygame.mixer.music.play(loops=0, start=int(self.scale.get()))
    
    def play_time(self):
        if self.stopped:
            return
        
        song_id = self.tree.selection()[0]
        song = self.tree.item(str(song_id),"values")[0]
        song = f'{self.playlist_path}/{song}.mp3'
        song_mut = MP3(song)
        self.song_length = song_mut.info.length
        song_length_cvt = time.strftime('%M:%S', time.gmtime(self.song_length))
        
        current_time = pygame.mixer.music.get_pos()/1000
        
        current_time_cvt = time.strftime('%M:%S', time.gmtime(current_time))
        current_time+=1
        
        if int(self.scale.get()) == int(self.song_length):
            self.elapse.configure(text="00:00")
            # self.next()
        elif self.paused:
            pass
        elif int(self.scale.get()) == int(current_time):
            slider_position = int(self.song_length)
            self.scale.config(to=slider_position, value=int(current_time))
        else:
            slider_position = int(self.song_length)
            self.scale.config(to=slider_position, value=int(self.scale.get()))
            current_time_cvt = time.strftime('%M:%S', time.gmtime(int(self.scale.get())))
            # self.remain.config(text=song_length_cvt)
            self.elapse.configure(text=current_time_cvt) 
            next_time = int(self.scale.get()) + 1
            self.scale.config(value=next_time)
        
        
        self.remain.config(text=song_length_cvt)
               
        self.elapse.after(1000, self.play_time)

if __name__ == '__main__':
    
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
    PATH='models/affectnet_emotions/enet_b0_5_ok.pt'
    # PATH = 'models/custom_model.pt'

    net_type = args.net_type

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
    new_frame_time = 0
    prev_frame_time = 0
    frame_checked = 30
    list_emo = []
    prev_emo = None
    # dict_emo = {
    #     'angry': 0, 'disgusted': 0, 'happy': 0, 'neutral': 0, 'sad': 0
    # }
    dict_emo = dict({'angry': 0, 'disgusted': 0, 'happy': 0, 'neutral': 0, 'sad': 0})
    cap = cv2.VideoCapture(cv2.CAP_DSHOW)  # capture from camera
    
    app = ttk.Window("Media Player", "yeti")
    
    mp = MediaPlayer(app)
    
    def show_frame():
        global dict_emo, sum, frame_count, new_frame_time, list_emo, prev_frame_time, frame_checked, prev_emo
        ret, orig_image = cap.read()
        frame_count += 1
        # if orig_image is None:
        #     print("end")
        #     break
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
            # im = ImageOps.grayscale(im)
            # im = im.resize((48,48))
            im = im.resize((224,224))
            
            scores = model(test_transforms(im).unsqueeze_(0))
            
            class_prob = torch.softmax(scores, dim=1)
            
            class_prob, topclass = torch.max(class_prob, dim=1)
            sco = class_prob.tolist()[0]
            
            state = idx_to_class[topclass.numpy()[0]]
            
            list_emo.append(state)
            if len(list_emo)>15:
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
            
            # scores=scores[0].data.cpu().numpy()
            # state = idx_to_class[np.argmax(scores[:5])]
            dict_emo[state] += 1
            
            if best_emo == "angry":
                s_emo = "tuc gian"
            elif best_emo =="disgusted":
                s_emo = "kho chiu"
            elif best_emo == "happy":
                s_emo = "vui ve"
            elif best_emo == "neutral":
                s_emo = "binh thuong"
            else:
                s_emo = "buon"
            cur_emo = s_emo 
            # best_emo = best_emo + " - " + "{:.3f}".format(sco)
            
            # cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[0])+aa, int(box[1])+aa), (0, 255, 0), 4)
            # cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 4)
            font = cv2.FONT_HERSHEY_SIMPLEX
            if frame_count %15==0 and frame_count >=15:
                cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 4)
                cv2.putText(orig_image,s_emo,(x+10,y+15), font, 0.5, (255,255,255), 2, cv2.LINE_AA)
                # cur_emo = best_emo 
            elif frame_count %15!=0 and frame_count >15:
                cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 4)
                cv2.putText(orig_image,cur_emo,(x+10,y+15), font, 0.5, (255,255,255), 2, cv2.LINE_AA)
            
            # cv2.putText(orig_image, fps, (7, 70), font, 1, (100, 255, 0), 1, cv2.LINE_AA)
            
        orig_image = cv2.resize(orig_image, None, None, fx=0.8, fy=0.8)
        sum += boxes.size(0)
        # cv2.imshow('Cam 0', orig_image)
        show_og_img = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        show_og_img = Image.fromarray(show_og_img)
        show_og_img = show_og_img.resize((800,600), Image.ANTIALIAS)
        imgtk = ImageTk.PhotoImage(image=show_og_img, width=725)
        mp.media.imgtk = imgtk
        mp.media.configure(image=imgtk)
        # mp.media.after(15, show_frame)
        
        positive = ['happy', 'neutral']
        negative = ['angry', 'disgusted', 'sad']
        if frame_count>=15:
            max_emo = best_emo
        if frame_checked == frame_count:
            
            
            print(f'max_emo: {max_emo}')
            
            prev_emo = max_emo
            print(f'prev emo: {prev_emo}')
            if max_emo in negative:
                playlist_p = "D:/VScode/linhtinh/playlist/playlist2"
            else:
                playlist_p = "D:/VScode/linhtinh/playlist/playlist1"
            song_length = mp.add_playlist(playlist_p)
            
        if frame_checked < frame_count:
            if int(mp.scale.get()) == int(mp.song_length):
                if max_emo != prev_emo:
                    mp.remove_playlist()
                    prev_emo = max_emo
                
                    if max_emo in negative:
                        playlist_p = "D:/VScode/linhtinh/playlist/playlist2"
                    else:
                        playlist_p = "D:/VScode/linhtinh/playlist/playlist1"
                    song_length = mp.add_playlist(playlist_p)
                    
                else:
                    mp.next()
                
            
        mp.media.after(15, show_frame)
    show_frame()
    
    mp.scale.set(3.35)  # set default
    app.mainloop()
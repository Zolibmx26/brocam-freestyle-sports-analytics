import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class VideoPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Player")

        self.video1_path = None
        self.video2_path = None
        self.video1_length = 0
        self.video2_length = 0
        self.playing = False

        self.canvas = tk.Canvas(self.root, width=800, height=400)
        self.canvas.grid(row=0, column=0, columnspan=2)

        self.btn_video1 = tk.Button(self.root, text="Select Video 1", command=self.select_video1)
        self.btn_video1.grid(row=1, column=0, padx=10, pady=5)

        self.btn_video2 = tk.Button(self.root, text="Select Video 2", command=self.select_video2)
        self.btn_video2.grid(row=1, column=1, padx=10, pady=5)

        self.timeline = tk.Scale(self.root, from_=0, to=100, orient=tk.HORIZONTAL, command=self.update_timeline)
        self.timeline.grid(row=2, column=0, columnspan=2, padx=10, pady=5)

        self.btn_play = tk.Button(self.root, text="Play", command=self.play)
        self.btn_play.grid(row=3, column=0, padx=10, pady=5)

        self.btn_stop = tk.Button(self.root, text="Stop", command=self.stop)
        self.btn_stop.grid(row=3, column=1, padx=10, pady=5)

        self.cap1 = None
        self.cap2 = None

    def select_video1(self):
        self.video1_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        self.play_video(self.video1_path, 0)

    def select_video2(self):
        self.video2_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        self.play_video(self.video2_path, 1)

    def play_video(self, video_path, video_num):
        cap = cv2.VideoCapture(video_path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if video_num == 0:
            self.video1_length = length
            self.cap1 = cap
        else:
            self.video2_length = length
            self.cap2 = cap

        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (400, 400))
            img = ImageTk.PhotoImage(image=Image.fromarray(frame))
            if video_num == 0:
                self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
                self.canvas.video1_img = img
            else:
                self.canvas.create_image(400, 0, anchor=tk.NW, image=img)
                self.canvas.video2_img = img

    def update_timeline(self, value):
        pass  # Implement timeline update logic

    def play(self):
        self.playing = True
        while self.playing:
            ret1, frame1 = self.cap1.read()
            ret2, frame2 = self.cap2.read()
            if ret1 and ret2:
                frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                frame1 = cv2.resize(frame1, (400, 400))
                img1 = ImageTk.PhotoImage(image=Image.fromarray(frame1))

                frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                frame2 = cv2.resize(frame2, (400, 400))
                img2 = ImageTk.PhotoImage(image=Image.fromarray(frame2))

                self.canvas.itemconfig(self.canvas.video1_img, image=img1)
                self.canvas.itemconfig(self.canvas.video2_img, image=img2)
                self.root.update()
            else:
                break

    def stop(self):
        self.playing = False
        self.cap1.release()
        self.cap2.release()


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoPlayer(root)
    root.mainloop()

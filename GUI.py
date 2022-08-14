from tkinter import *
from PIL import Image, ImageTk
import PIL
from tkinter.filedialog import askdirectory
from tkinter import ttk
import os
import time
import matplotlib.pyplot as plt
import cv2
import skimage.io

top = Tk()
top.title('智能识别系统')

top.geometry('650x625')
top.resizable(width=True, height=True)
top.maxsize(650, 625)
top.minsize(650, 625)

path = StringVar()

def predictimage():
    import os
    import sys
    import skimage.io
    from mrcnn.config import Config
    from datetime import datetime
    import mrcnn.model as modellib
    from visualize import display_instances
    from mrcnn import visualize

    ROOT_DIR = os.getcwd()
    # Import Mask RCNN
    sys.path.append(ROOT_DIR)  # To find local version of the library
    from mrcnn import utils
    # Import COCO config
    sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Local path to trained weights file
    # COCO_MODEL_PATH = ""
    # if operate == 0:
    COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_shapes_guangfu_0100.h5")
    # elif operate == 1:
    #     COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_shapes_rongchi_0100.h5")
    # # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        # utils.download_trained_weights(COCO_MODEL_PATH)
        print("找不到权重文件")

    # Directory of images to run detection on

    class ShapesConfig(Config):
        """Configuration for training on the toy shapes dataset.
        Derives from the base Config class and overrides values specific
        to the toy shapes dataset.
        """
        # Give the configuration a recognizable name
        NAME = "shapes"

        # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
        # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
        GPU_COUNT = 1
        IMAGES_PER_GPU = 2

        # Number of classes (including background)
        # if operate == 0:
        NUM_CLASSES = 1 + 3  # background + 3 shapes
        # elif operate == 1:
        #     NUM_CLASSES = 1 + 1

        # Use small images for faster training. Set the limits of the small side
        # the large side, and that determines the image shape.
        IMAGE_MIN_DIM = 320
        IMAGE_MAX_DIM = 384

        # Use smaller anchors because our image and objects are small
        RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels

        # Reduce training ROIs per image because the images are small and have
        # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
        TRAIN_ROIS_PER_IMAGE = 100

        # Use a small epoch since the data is simple
        STEPS_PER_EPOCH = 100

        # use small validation steps since the epoch is small
        VALIDATION_STEPS = 50

    # import train_tongue
    # class InferenceConfig(coco.CocoConfig):
    class InferenceConfig(ShapesConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()

    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')
    class_names =[]
    # if operate == 0:
    class_names = ['board', 'person', 'plant']
    # elif operate == 1:
    #     class_names = ['molten_pool']
    # Load a random image from the images folder

    # 自己输入想要预测的图片
    file_names = numberChosen1.get()
    global image
    image = skimage.io.imread(ROOT_DIR+'/'+path_for_jpg+'/'+str(file_names))

    a = datetime.now()
    # Run detection
    results = model.detect([image], verbose=1)
    b = datetime.now()
    # Visualize results
    r = results[0]
    print(r['scores'])
    img = display_instances(image, r['rois'], r['masks'], r['class_ids'] - 1, class_names, r['scores'])
    #lmain11.destroy()
    lmain0.destroy()
    global lmain1
    # if operate == 0:
    lmain1 = Label(canvas, width=590, height=480)
    # elif operate == 1:
    #     lmain1 = Label(canvas, width=590, height=480)
    lmain1.pack()
    img = Image.fromarray(img).resize((590, 480), Image.BILINEAR)
    img = ImageTk.PhotoImage(img)
    lmain1.imgtk = img
    lmain1.configure(image=img)
    print("时间:", (b - a).seconds)

def predictvideo():
    # global lmain22
    # lmain22 = Label(canvas, width=590, height=480)
    # lmain22.pack()
    # print(r"E:\maskrcnn\Mask_RCNN_master\images\光伏\3.jpg")
    # img_temp = skimage.io.imread(r"E:\maskrcnn\Mask_RCNN_master\images\光伏\3.jpg")
    # img_temp = Image.fromarray(img_temp).resize((590, 480), Image.BILINEAR)
    # img_tk = ImageTk.PhotoImage(img_temp)
    # lmain22.configure(image=img_tk)
    # lmain22.image = img_tk
    # print("展示图片")
    # -*- coding: utf-8 -*-
    import glob
    import os
    import sys
    import cv2
    from mrcnn.config import Config
    from visualize import display_instances, random_colors
    from mrcnn import utils
    import mrcnn.model as modellib
    ROOT_DIR = os.getcwd()
    sys.path.append(ROOT_DIR)  # To find local version of the library
    # Import COCO config
    sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
    from samples.coco import coco

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Local path to trained weights file
    # COCO_MODEL_PATH = ""
    # if operate == 0:
    COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_shapes_guangfu_0100.h5")
    # elif operate == 1:
    #     COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_shapes_rongchi_0100.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        #utils.download_trained_weights(COCO_MODEL_PATH)
        print("找不到权重文件")

    # Directory of images to run detection on

    class ShapesConfig(Config):
        """Configuration for training on the toy shapes dataset.
        Derives from the base Config class and overrides values specific
        to the toy shapes dataset.
        """
        # Give the configuration a recognizable name
        NAME = "shapes"

        # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
        # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
        GPU_COUNT = 1
        IMAGES_PER_GPU = 2

        # Number of classes (including background)
        # if operate == 0:
        NUM_CLASSES = 1 + 3  # background + 3 shapes
        # elif operate == 1:
        #     NUM_CLASSES = 1 + 1

        # Use small images for faster training. Set the limits of the small side
        # the large side, and that determines the image shape.
        IMAGE_MIN_DIM = 320
        IMAGE_MAX_DIM = 384

        # Use smaller anchors because our image and objects are small
        RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels

        # Reduce training ROIs per image because the images are small and have
        # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
        TRAIN_ROIS_PER_IMAGE = 100

        # Use a small epoch since the data is simple
        STEPS_PER_EPOCH = 100

        # use small validation steps since the epoch is small
        VALIDATION_STEPS = 50

    # import train_tongue
    # class InferenceConfig(coco.CocoConfig):
    class InferenceConfig(ShapesConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()

    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')
    # class_names = []
    # if operate == 0:
    class_names = ['board', 'person', 'plant']
    # elif operate == 1:
    #     class_names = ['molten_pool']
    # Root directory of the project
    video = numberChosen2.get()
    video_path = os.path.join(ROOT_DIR+'/', str(video))
    capture = cv2.VideoCapture(video_path)  # 这里是输入视频的文件名
    VIDEO_SAVE_DIR = os.path.join(ROOT_DIR, 'save_pic')

    if not os.path.exists(VIDEO_SAVE_DIR+'/'+video):
        print(video_path)
        print(VIDEO_SAVE_DIR)
        frames = []
        frame_count = 0
        # these 2 lines can be removed if you dont have a 1080p camera.
        # capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        batch_size = 1
        colors = random_colors(len(class_names))
        while True:
            ret, frame = capture.read()
            # Bail out when the video file ends
            if not ret:
                print("1")
                break

            # Save each frame of the video to a list
            frame_count += 1
            frames.append(frame)
            print('frame_count :{0}'.format(frame_count))
            if len(frames) == batch_size:
                results = model.detect(frames, verbose=0)
                print('Predicted')
                for i, item in enumerate(zip(frames, results)):
                    frame = item[0]
                    r = item[1]
                    frame = display_instances(frame, r['rois'], r['masks'], r['class_ids'] - 1, class_names, r['scores'])
                    name = '{0}.jpg'.format(frame_count + i - batch_size)
                    name = os.path.join(VIDEO_SAVE_DIR, name)
                    cv2.imwrite(name, frame)
                    print('writing to file:{0}'.format(name))
                # Clear the frames array to start the next batch
                frames = []

        capture.release()

        def make_video(outvid, images=None, fps=30, size=None,
                       is_color=True, format="FMP4"):

            from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
            fourcc = VideoWriter_fourcc(*format)
            vid = None
            for image in images:
                print(image)
                if not os.path.exists(image):
                    raise FileNotFoundError(image)
                img = imread(image)
                if vid is None:
                    if size is None:
                        size = img.shape[1], img.shape[0]
                    vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
                if size[0] != img.shape[1] and size[1] != img.shape[0]:
                    img = resize(img, size)
                vid.write(img)
            # vid.release()
            return vid

        def removeFileInDir(sourceDir):
            for file in os.listdir(sourceDir):
                file = os.path.join(sourceDir, file)  # 必须拼接完整文件名
                if os.path.isfile(file) and file.find(".jpg") > 0:
                    os.remove(file)
                    print(file + " remove succeeded")

        # Directory of images to run detection on
        images_dir = os.path.join(ROOT_DIR, "save_pic")
        images = list(glob.iglob(os.path.join(images_dir, '*.*')))

        # Sort the images by integer index
        images = sorted(images, key=lambda x: float(os.path.split(x)[1][:-3]))
        outvid = os.path.join(images_dir, video)
        make_video(outvid, images, fps=30)
        removeFileInDir(VIDEO_SAVE_DIR)
        print('make video success')

    #lmain22.destroy()
    lmain0.destroy()
    global lmain2,capture1
    # if operate == 0:
    lmain2 = Label(canvas, width=590, height=480)
    # elif operate == 1:
    #     lmain2 = Label(canvas, width=590, height=480)
    lmain2.pack()
    capture1 = cv2.VideoCapture('save_pic/'+video)
    def video_stream():
        _,frame = capture1.read()
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain2.imgtk = imgtk
        lmain2.configure(image=imgtk)
        lmain2.after(1, video_stream)
    video_stream()

def predictcamera():
    import os
    import sys
    import random
    import math
    import numpy as np
    import skimage.io
    import matplotlib
    import matplotlib.pyplot as plt
    import cv2
    import time
    from mrcnn.config import Config
    from datetime import datetime
    # Root directory of the project
    ROOT_DIR = os.getcwd()

    # Import Mask RCNN
    sys.path.append(ROOT_DIR)  # To find local version of the library
    from mrcnn import utils
    import mrcnn.model as modellib
    from visualize import display_instances

    # Import COCO config
    sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
    from samples.coco import coco

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Local path to trained weights file
    # COCO_MODEL_PATH = ""
    # if operate == 0:
    COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_shapes_guangfu_0100.h5")
    # elif operate == 1:
    #     COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_shapes_rongchi_0100.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        print("找不到权重文件")

    # Directory of images to run detection on
    IMAGE_DIR = os.path.join(ROOT_DIR, "images")

    class ShapesConfig(Config):
        """Configuration for training on the toy shapes dataset.
        Derives from the base Config class and overrides values specific
        to the toy shapes dataset.
        """
        # Give the configuration a recognizable name
        NAME = "shapes"

        # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
        # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
        GPU_COUNT = 1
        IMAGES_PER_GPU = 2

        # Number of classes (including background)
        # if operate == 0:
        NUM_CLASSES = 1 + 3  # background + 3 shapes
        # elif operate == 1:
        #     NUM_CLASSES = 1 + 1

        # Use small images for faster training. Set the limits of the small side
        # the large side, and that determines the image shape.
        IMAGE_MIN_DIM = 320
        IMAGE_MAX_DIM = 384

        # Use smaller anchors because our image and objects are small
        RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels

        # Reduce training ROIs per image because the images are small and have
        # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
        TRAIN_ROIS_PER_IMAGE = 100

        # Use a small epoch since the data is simple
        STEPS_PER_EPOCH = 100

        # use small validation steps since the epoch is small
        VALIDATION_STEPS = 50

    # import train_tongue
    # class InferenceConfig(coco.CocoConfig):
    class InferenceConfig(ShapesConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()

    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')
    # class_names = []
    # if operate == 0:
    class_names = ['board', 'person', 'plant']
    # elif operate == 1:
    #     class_names = ['molten_pool']
    # 摄像头识别
    lmain0.destroy()
    global lmain3,capture2
    # if operate == 0:
    lmain3 = Label(canvas, width=590, height=480)
    # elif operate == 1:
    #     lmain3 = Label(canvas, width=590, height=480)
    lmain3.pack()
    capture2 = cv2.VideoCapture(0)
    def video_stream():
        ret, frame = capture2.read()
        results = model.detect([frame], verbose=0)
        r = results[0]
        frame = display_instances(
            frame, r['rois'], r['masks'], r['class_ids'] - 1,
            class_names, r['scores']
        )
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain3.imgtk = imgtk
        lmain3.configure(image=imgtk)
        lmain3.after(1, video_stream)

    video_stream()

# def imageshow(img_name_):
#     global lmain11
#     lmain11 = Label(canvas, width=590, height=480)
#     lmain11.pack()
#     print(img_name_)
#     img_temp = skimage.io.imread(img_name_)
#     img_temp = Image.fromarray(img_temp).resize((590, 480), Image.BILINEAR)
#     img_tk = ImageTk.PhotoImage(img_temp)
#     lmain11.configure(image=img_tk)
#     lmain11.image = img_tk
#     print("展示图片")

def select_path():

    global path_for_jpg,path_for_video
    path_ = askdirectory()
    path.set(path_)
    path_for_jpg = path_+"/"
    path_for_video = path_ + "/"
    print("path is :" + str(path_for_jpg).replace(' ', ''))
    print("path is :" + str(path_for_video).replace(' ', ''))

def find_jpg(path1):

    print("读取文件")
    global h
    f = []
    for root, dirs, files in os.walk(path1):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                t = os.path.splitext(file)
                s = str(t[0])+str(t[1])
                print(s)  # 打印所有py格式的文件名
                f.append(s)  # 将所有的文件名添加到F列表中
    h = f
    numberChosen1['values'] = h  # 设置下拉列表的值

def find_video(path1):

    print("读取文件")
    global h
    f = []
    for root, dirs, files in os.walk(path1):
        for file in files:
            if os.path.splitext(file)[1] == '.mp4':
                t = os.path.splitext(file)
                s = str(t[0])+str(t[1])
                print(s)  # 打印所有py格式的文件名
                f.append(s)  # 将所有的文件名添加到F列表中
    h = f
    numberChosen2['values'] = h  # 设置下拉列表的值

def clossimage():
    lmain1.destroy()
    global lmain0
    lmain0 = Label(canvas, width=590, height=480)
    lmain0.pack()
    img_temp = skimage.io.imread(r'D:\HLW+\Mask_RCNN_master\Mask_RCNN_master\images\初始图片.jpg')
    img_temp = Image.fromarray(img_temp).resize((590, 480), Image.BILINEAR)
    img_tk = ImageTk.PhotoImage(img_temp)
    lmain0.configure(image=img_tk)
    lmain0.image = img_tk
def clossvideo():
    lmain2.destroy()
    capture1.release()
    global lmain0
    lmain0 = Label(canvas, width=590, height=480)
    lmain0.pack()
    img_temp = skimage.io.imread(r'D:\HLW+\Mask_RCNN_master\Mask_RCNN_master\images\初始图片.jpg')
    img_temp = Image.fromarray(img_temp).resize((590, 480), Image.BILINEAR)
    img_tk = ImageTk.PhotoImage(img_temp)
    lmain0.configure(image=img_tk)
    lmain0.image = img_tk
def closecamera():
    lmain3.destroy()
    capture2.release()
    global lmain0
    lmain0 = Label(canvas, width=590, height=480)
    lmain0.pack()
    img_temp = skimage.io.imread(r'D:\HLW+\Mask_RCNN_master\Mask_RCNN_master\images\初始图片.jpg')
    img_temp = Image.fromarray(img_temp).resize((590, 480), Image.BILINEAR)
    img_tk = ImageTk.PhotoImage(img_temp)
    lmain0.configure(image=img_tk)
    lmain0.image = img_tk
path_for_jpg = "images/光伏"
path_for_video = "video/光伏"
# def change1():
#     global operate,path_for_jpg,path_for_video
#     operate = 0
#     path_for_jpg = "images/光伏"
#     path_for_video = "video/光伏"
# def change2():
#     global operate,path_for_jpg,path_for_video
#     operate = 1
#     path_for_jpg = "images/熔池"
#     path_for_video = "video/熔池"


canvas = Canvas(top, bg='white', width=590, height=480)
canvas.pack()
global lmain0,img_tk
lmain0 = Label(canvas, width=590, height=480)
lmain0.pack()
img_temp = skimage.io.imread(r'D:\HLW+\Mask_RCNN_master\Mask_RCNN_master\images\初始图片.jpg')
img_temp = Image.fromarray(img_temp).resize((590, 480), Image.BILINEAR)
img_tk = ImageTk.PhotoImage(img_temp)
lmain0.configure(image=img_tk)
lmain0.image = img_tk


# global label
# label = Label(canvas, width=590, height=1)
# label.pack()


button_get_file_in_dir1 = Button(top, text = "读取", command = lambda:find_jpg(path1=path_for_jpg))
number1 = StringVar()
numberChosen1 = ttk.Combobox(top, width=12, textvariable=number1)
numberChosen1['values'] = ["请点击以下图片进行识别"]    # 设置下拉列表的值
numberChosen1.current(0)
button_get_file_in_dir1.place(x=30, y=510)
numberChosen1.place(x=90,y=510)

button_get_file_in_dir2 = Button(top, text = "读取", command = lambda:find_video(path1=path_for_video))
number2 = StringVar()
numberChosen2 = ttk.Combobox(top, width=12, textvariable=number2)
numberChosen2['values'] = ["请点击视频名字识别"]    # 设置下拉列表的值
numberChosen2.current(0)
button_get_file_in_dir2.place(x=230, y=510)
numberChosen2.place(x=290,y=510)


menubar = Menu(top)
filemenu = Menu(menubar, tearoff=0)

#添加菜单栏
# menubar.add_cascade(label='请先选择识别类型', menu=filemenu)
# filemenu.add_command(label='光伏类型', command=change1)
# filemenu.add_separator()#添加横线
# filemenu.add_command(label='溶池类型', command=change2)
# top.config(menu=menubar)

# button0 = Button(top, text="展示图片", activeforeground='red', relief=RIDGE, command = lambda:imageshow(img_name_=path_for_jpg+'/'+numberChosen1.get()))
# button0.place(x=120, y=525)
button1 = Button(top, text="识别图片", activeforeground='red', relief=RIDGE, command=predictimage)
button1.place(x=30, y=565)
button3 = Button(top, text="关闭图片", activeforeground='red', relief=RIDGE, command=clossimage)
button3.place(x=120, y=565)
button2 = Button(top, text="识别视频", activeforeground='red', relief=RIDGE, command=predictvideo)
button2.place(x=230, y=565)
button3 = Button(top, text="关闭视频", activeforeground='red', relief=RIDGE, command=clossvideo)
button3.place(x=320, y=565)
button3 = Button(top, text="打开摄像头", activeforeground='red', relief=RIDGE, command=predictcamera)
button3.place(x=430, y=510)
button3 = Button(top, text="关闭摄像头", activeforeground='red', relief=RIDGE, command=closecamera)
button3.place(x=430, y=565)
button4 = Button(top, text="退出", activeforeground='red', relief=RIDGE, command=top.quit, height=4, width=10)
button4.place(x=540, y=510)
top.mainloop()
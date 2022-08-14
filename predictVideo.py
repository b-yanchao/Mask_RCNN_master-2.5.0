# -*- coding: utf-8 -*-

import glob
import os
import sys
import cv2
from mrcnn.config import Config
from visualize import display_instances,random_colors
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
COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_shapes_guangfu_0100.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
    print("cuiwei***********************")

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
    NUM_CLASSES = 1 + 3  # background + 3 shapes

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
class_names = ['board','person','plant']


# Root directory of the project
video_path = os.path.join(ROOT_DIR, 'images/0.mp4')
capture = cv2.VideoCapture(video_path) #这里是输入视频的文件名
VIDEO_SAVE_DIR = os.path.join(ROOT_DIR, 'save_pic')
print(video_path)
print(VIDEO_SAVE_DIR)
frames = []
frame_count = 0
# these 2 lines can be removed if you dont have a 1080p camera.
#capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
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
            frame = display_instances(frame, r['rois'], r['masks'], r['class_ids']-1, class_names, r['scores'])
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
        file=os.path.join(sourceDir,file) #必须拼接完整文件名
        if os.path.isfile(file) and file.find(".jpg")>0:
            os.remove(file)
            print(file+" remove succeeded")

# Directory of images to run detection on
images_dir = os.path.join(ROOT_DIR, "save_pic")
images = list(glob.iglob(os.path.join(images_dir, '*.*')))

# Sort the images by integer index
images = sorted(images, key=lambda x: float(os.path.split(x)[1][:-3]))

outvid = os.path.join(images_dir, "detect_out.mp4")
make_video(outvid,images, fps=30)
removeFileInDir(VIDEO_SAVE_DIR)
print('make video success')


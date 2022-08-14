import os
import sys
import cv2
import tkinter
from PIL import Image, ImageTk
from GUI import canvas
import time
from mrcnn.config import Config
def x1():
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
    COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_shapes_0031.h5")
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
    class_names = ['borad', 'person', 'plant']
    # # class_names = ['molten_pool']
    # 摄像头识别
    lmain = tkinter.Label(canvas, width=450, height=400)
    lmain.pack()
    capture = cv2.VideoCapture(0)
    def video_stream():
        ret, frame = capture.read()
        results = model.detect([frame], verbose=0)
        r = results[0]
        frame = display_instances(
            frame, r['rois'], r['masks'], r['class_ids'] - 1,
            class_names, r['scores']
        )
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(1, video_stream)

    video_stream()
if __name__ == '__main__':
    x1()
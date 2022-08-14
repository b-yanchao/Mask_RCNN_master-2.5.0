import numpy as np
import cv2
import os
import json
def SaltAndPepper(src, percetage):
    SP_NoiseImg = src.copy()
    SP_NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(SP_NoiseNum):
        randR = np.random.randint(0, src.shape[0] - 1)
        randG = np.random.randint(0, src.shape[1] - 1)
        randB = np.random.randint(0, 3)
        if np.random.randint(0, 1) == 0:
            SP_NoiseImg[randR, randG, randB] = 0
        else:
            SP_NoiseImg[randR, randG, randB] = 255
    return SP_NoiseImg


def addGaussianNoise(image, percetage):
    G_Noiseimg = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    G_NoiseNum = int(percetage * image.shape[0] * image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(0, h)
        temp_y = np.random.randint(0, w)
        G_Noiseimg[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0]
    return G_Noiseimg


# dimming
def darker(image, percetage=0.9):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get darker
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = int(image[xj, xi, 0] * percetage)
            image_copy[xj, xi, 1] = int(image[xj, xi, 1] * percetage)
            image_copy[xj, xi, 2] = int(image[xj, xi, 2] * percetage)
    return image_copy


def brighter(image, percetage=1.5):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get brighter
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = np.clip(int(image[xj, xi, 0] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 1] = np.clip(int(image[xj, xi, 1] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 2] = np.clip(int(image[xj, xi, 2] * percetage), a_max=255, a_min=0)
    return image_copy


def rotate(image, angle=15, scale=0.9):
    w = image.shape[1]
    h = image.shape[0]
    # rotate matrix
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    # rotate
    image = cv2.warpAffine(image, M, (w, h))
    return image


def img_augmentation(path):
    imglist = os.listdir(path)
    for i in range(len(imglist)):
        imgpath = os.path.join(path, imglist[i])
        img = cv2.imread(imgpath)
        img_flip = cv2.flip(img, 1)  # flip翻转
        img_rotation = rotate(img)  # rotation
        img_noise1 = SaltAndPepper(img, 0.3)
        img_noise2 = addGaussianNoise(img, 0.3)
        img_brighter = brighter(img)
        img_darker = darker(img)
        rename_path_img_flip = os.path.join(path,'flip'+ imglist[i])
        rename_path_img_rotate = os.path.join(path,'rotate'+ imglist[i])
        rename_path_img_SaltAndPepper = os.path.join(path, 'SaltAndPepper' + imglist[i])
        rename_path_img_addGaussianNoise = os.path.join(path, 'addGaussianNoise' + imglist[i])
        rename_path_img_brighter = os.path.join(path, 'brighter' + imglist[i])
        rename_path_img_darker = os.path.join(path, 'darker' + imglist[i])

        cv2.imwrite(rename_path_img_flip, img_flip)
        cv2.imwrite(rename_path_img_rotate, img_rotation)
        cv2.imwrite(rename_path_img_SaltAndPepper, img_noise1)
        cv2.imwrite(rename_path_img_addGaussianNoise, img_noise2)
        cv2.imwrite(rename_path_img_brighter, img_brighter)
        cv2.imwrite(rename_path_img_darker, img_darker)


def save_json(file_name, save_folder, dic_info):
    with open(os.path.join(save_folder, file_name), 'w') as f:
        json.dump(dic_info, f, indent=2)
img_augmentation('train_dataset\pic')
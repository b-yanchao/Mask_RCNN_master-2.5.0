import cv2
import numpy as np
# 定义随机颜色函数
def random_colors(N):
    np.random.seed(1)
    colors=[tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors

def apply_mask(image,mask,color,alpha=1):
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image


def display_instances(image, boxes, masks, ids, names, scores):
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0]
    colors = random_colors(n_instances)

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        score = scores[i] if scores is not None else None
        caption = "{} {:.3f}".format(label, score) if score else label
        mask = masks[:, :, i]

        image = apply_mask(image, mask, (255, 0, 0))
        #image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        #image = cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255))

    return image
#!/usr/bin/env python
# coding: utf-8

# In[18]:


from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

get_ipython().run_line_magic("matplotlib", "inline")
import cv2


def multiply(tup_):
    result = 1
    for i in tup_:
        result += i
    return result


import pytesseract
import warnings

warnings.filterwarnings("ignore")


# In[2]:


model = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path="../best.pt",
    force_reload=True,
)
_ = model.eval()


# In[3]:


img_path = "example.jpg"
save_path = "result.jpg"

img = Image.open(img_path)
img_arr = np.array(img)
result = model(img_arr)


# In[ ]:


res = result.pred[0].detach()


# In[19]:


fig = plt.figure(frameon=False)
ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
ax.set_axis_off()
fig.add_axes(ax)

final_res = []
for t in res:
    x1, y1, x2, y2, proba, _ = t.cpu().numpy()
    if proba > 0.6:
        ax.imshow(img)
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="r", facecolor="none"
        )
        ax.add_patch(rect)
        final_res.append([x1, y1, x2, y2])
_ = fig.savefig(save_path, bbox_inches="tight", pad_inches=0)


# In[21]:


for num_plate in final_res:
    plate_img = img.crop(num_plate)

    im_gray = cv2.cvtColor(np.array(plate_img).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    th, im_gray_th_otsu = cv2.threshold(im_gray, 128, 255, cv2.THRESH_OTSU)

    SCALE_FACTOR = 1

    plate_img = plate_img.resize(
        [i * SCALE_FACTOR for i in Image.fromarray(im_gray_th_otsu).size]
    )

    cnts = cv2.findContours(im_gray_th_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sizes_of_0_cnts = [multiply(i.shape) for i in cnts[0]]
    cnt = cnts[0][np.argmax(sizes_of_0_cnts)]

    xmin, xmax, ymin, ymax = (
        min(cnt[:, 0, 0]),
        max(cnt[:, 0, 0]),
        min(cnt[:, 0, 1]),
        max(cnt[:, 0, 1]),
    )

    x1, y1 = cnt[np.argmax(cnt[:, 0, 0]), 0, :]
    x2, y2 = cnt[np.argmin(cnt[:, 0, 0]), 0, :]
    x3, y3 = cnt[np.argmax(cnt[:, 0, 1]), 0, :]
    x4, y4 = cnt[np.argmin(cnt[:, 0, 1]), 0, :]

    fig, ax = plt.subplots()

    ax.imshow(plate_img, cmap="gray")
    rect = patches.Rectangle(
        (x1, y1), 1, 1, linewidth=5, edgecolor="r", facecolor="r", label="1"
    )
    ax.add_patch(rect)
    rect = patches.Rectangle(
        (x2, y2), 1, 1, linewidth=5, edgecolor="g", facecolor="g", label="2"
    )
    ax.add_patch(rect)
    rect = patches.Rectangle(
        (x3, y3), 1, 1, linewidth=5, edgecolor="b", facecolor="b", label="3"
    )
    ax.add_patch(rect)
    rect = patches.Rectangle(
        (x4, y4), 1, 1, linewidth=5, edgecolor="orange", facecolor="orange", label="4"
    )
    ax.add_patch(rect)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.show()

    tg_alpha_32 = abs(y3 - y2) / abs(x3 - x2)
    tg_alpha_41 = abs(y4 - y1) / abs(x4 - x1)
    tg_alpha = (tg_alpha_32 + tg_alpha_41) / 2
    new_plate_img = Image.fromarray(im_gray_th_otsu).rotate(
        (np.arctan(tg_alpha) * 180) / np.pi, Image.NEAREST, expand=1
    )

    new_plate_img

    h, w = np.array(new_plate_img).shape

    np.argwhere(np.array(new_plate_img).sum(axis=1) != 0).flatten()

    h_idxs = np.argwhere(np.array(new_plate_img).sum(axis=1) != 0).flatten()
    w_idxs = np.argwhere(np.array(new_plate_img).sum(axis=0) != 0).flatten()

    plate_arr = np.array(new_plate_img)[h_idxs[0] : h_idxs[-1], w_idxs[0] : w_idxs[-1]]

    custom_config = r"-c tessedit_char_whitelist=0123456789ABEKMHOPCTYX --oem 3 --psm 6"
    recognized = pytesseract.image_to_string(plate_arr, config=custom_config)
    print("Recognized number:", recognized)


# In[ ]:

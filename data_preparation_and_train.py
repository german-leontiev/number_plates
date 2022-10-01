#!/usr/bin/env python
# coding: utf-8

# In[50]:


from pathlib import Path as p
from glob import glob
import json
import shutil as sh
import warnings
from tqdm import tqdm
from PIL import Image as I
from matplotlib import pyplot as plt
import imagesize

warnings.filterwarnings("ignore")


# In[52]:


dataset_root_dir = "custom_dataset"
p.mkdir(p(dataset_root_dir), exist_ok=True)
sh.rmtree(dataset_root_dir)
train_images_dir = "images/train"
val_images_dir = "images/val"
test_images_dir = "images/test"

p.mkdir(p(dataset_root_dir), exist_ok=True, parents=True)

for path in [train_images_dir, val_images_dir, test_images_dir]:
    p.mkdir(p(dataset_root_dir) / path, exist_ok=True, parents=True)

yaml_content = f"""
path: ../{dataset_root_dir} 
train: {train_images_dir} 
val: {val_images_dir}
test:  {test_images_dir}
names:
  0: dummy
  1: plate
"""

with open("custom_dataset.yaml", "w") as f:
    f.write(yaml_content)


# In[53]:


source_dataset = "nomeroff_datasets/autoriaNumberplateDataset-2022-08-01/"
general_nr = {"train": [], "val": []}
for subset in ["train", "val"]:
    all_images_paths = [
        path
        for path in glob(
            f"nomeroff_datasets/autoriaNumberplateDataset-2022-08-01/{subset}/*"
        )
        if ".json" not in path and ".sh" not in path
    ]
    annotations_path = [
        path
        for path in glob(
            f"nomeroff_datasets/autoriaNumberplateDataset-2022-08-01/{subset}/*"
        )
        if ".json" in path
    ][0]
    with open(annotations_path, "r") as f:
        annotation = json.loads(f.read())
    #     ann_dir = dataset_root_dir + "/" + f"labels/{subset}" + "/"

    #     p.mkdir(p(ann_dir), parents=True, exist_ok=True)

    nr = [
        (fn["filename"], fn["regions"])
        for fn in list(annotation["_via_img_metadata"].values())
    ]
    general_nr[subset] = nr

#     for filename, regions in tqdm(nr):

#         s = source_dataset + subset + "/" + filename
#         d = dataset_root_dir + "/" + f"images/{subset}" + "/" + filename
#         sh.copy(s, d)
#         image_width, image_height = imagesize.get(s)
#         ann_content = ""
#         for region in regions:
#             shape_attributes = region["shape_attributes"]
#             if shape_attributes["name"] == "polygon":
#                 all_points_x = region["shape_attributes"]["all_points_x"]
#                 all_points_y = region["shape_attributes"]["all_points_y"]
#                 width = (max(all_points_x) - min(all_points_x)) / image_width
#                 height = (max(all_points_y) - min(all_points_y)) / image_height

#                 x_center = min(all_points_x) / image_width + width / 2
#                 y_center = min(all_points_y) / image_height + height / 2

#                 ann_content += "1 {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
#                     x_center, y_center, width, height
#                 )

#         basename = ".".join(filename.split(".")[:-1])
#         txt_save_path = ann_dir + basename + ".txt"

#         with open(txt_save_path, "w") as f:
#             f.write(ann_content)


# In[54]:


TOTAL_IMAGES = len(general_nr["val"]) + len(general_nr["train"])


# In[55]:


nr = {}


# In[56]:


nr["val"] = general_nr["val"]
nr["test"] = general_nr["train"][-int(TOTAL_IMAGES * 0.1) :]
nr["train"] = general_nr["train"][: -int(TOTAL_IMAGES * 0.1)]


# In[57]:


for subset in nr.keys():
    for filename, regions in tqdm(nr[subset]):
        if subset == "test":
            source_subset = "train"
        else:
            source_subset = subset

        ann_dir = dataset_root_dir + "/" + f"labels/{subset}" + "/"

        p.mkdir(p(ann_dir), parents=True, exist_ok=True)
        s = source_dataset + source_subset + "/" + filename
        d = dataset_root_dir + "/" + f"images/{subset}" + "/" + filename
        sh.copy(s, d)
        image_width, image_height = imagesize.get(s)
        ann_content = ""
        for region in regions:
            shape_attributes = region["shape_attributes"]
            if shape_attributes["name"] == "polygon":
                all_points_x = region["shape_attributes"]["all_points_x"]
                all_points_y = region["shape_attributes"]["all_points_y"]
                width = (max(all_points_x) - min(all_points_x)) / image_width
                height = (max(all_points_y) - min(all_points_y)) / image_height

                x_center = min(all_points_x) / image_width + width / 2
                y_center = min(all_points_y) / image_height + height / 2

                ann_content += "1 {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                    x_center, y_center, width, height
                )

        basename = ".".join(filename.split(".")[:-1])
        txt_save_path = ann_dir + basename + ".txt"

        with open(txt_save_path, "w") as f:
            f.write(ann_content)


# # Эксперименты
# Поскольку у нас ограниченное число времени ограничимся ограниченным числом параметров и моделей. Всего 27 тестов займут очень солидное время

# In[58]:


model_types = ["yolov5n", "yolov5m", "yolov5x"]
epochs = [20, 30, 40]
optimizers = ["SGD", "Adam", "AdamW"]


# In[ ]:


for mt in model_types:
    for e in epochs:
        for opt in optimizers:
            name = f"{mt}_{opt}_{e}"
            weights = mt + ".pt"
            get_ipython().system(
                ' python yolov5/train.py --img 640 --batch 4 --epochs $e --data custom_dataset.yaml --weights $weights --project "tests" --name $name'
            )


# In[ ]:

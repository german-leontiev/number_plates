#!/usr/bin/env python
# coding: utf-8

# In[8]:


from glob import glob
import pandas as pd
import shutil


# Сравним все модели по метрике box_loss на валидационной выборке и выберем лучший вариант

# In[3]:


best_loss = 1
best_model = None
for i, test in enumerate(glob("tests/*")):
    res_path = test + "/results.csv"

    df_res = pd.read_csv(res_path)

    val_box_col = [l for l in df_res.columns if "val/box_loss" in l][0]

    last_box_loss = df_res.loc[:,val_box_col].values[-1]
    model, optimizer, n_epochs = test.split("/")[1].split("_")
    print(f"Model: {model}\nOptimizer: {optimizer}\nNumber of epochs: {n_epochs}\n\nBox_loss: {last_box_loss}\n" + "="*10)
    if last_box_loss < best_loss:
        best_loss = last_box_loss
        best_model = test


# Скопируем модель в корень проекта с именем `best.pt`

# In[7]:


shutil.copy(best_model + '/weights/best.pt', "best.pt")


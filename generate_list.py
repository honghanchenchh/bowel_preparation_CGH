import os
import random

classes=['clean','feces']

data_dir='/home/hogan/data1/chh/clean_exp/EasyDL_data'

all_list=[]
for cls in classes:
    cls_folder=os.path.join(data_dir,cls)
    files=os.listdir(cls_folder)
    for fi in files:
        if not fi.endswith('.jpg'):
            continue
        tmp_file=os.path.join(cls_folder,fi)
        tmp_line=tmp_file+'\t'+str(classes.index(cls))
        all_list.append(tmp_line)



random.shuffle(all_list)
with open('/home/hogan/data1/chh/clean_exp/code_classification/shuffle_3.txt','w') as f:
    for itm in all_list:
        f.write(itm+'\n')
train_len=int(len(all_list)*0.8)
val_len=int(len(all_list)*0.1)

train_list=all_list[:train_len]
val_list=all_list[train_len:train_len+val_len]
test_list=all_list[train_len+val_len:]
with open('/home/hogan/data1/chh/clean_exp/code_classification/train_3.txt','w') as f:
    for itm in train_list:
        f.write(itm+'\n')

with open('/home/hogan/data1/chh/clean_exp/code_classification/val_3.txt','w') as f:
    for itm in val_list:
        f.write(itm+'\n')

with open('/home/hogan/data1/chh/clean_exp/code_classification/test_3.txt','w') as f:
    for itm in test_list:
        f.write(itm+'\n')




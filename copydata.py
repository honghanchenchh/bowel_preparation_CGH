import shutil
import os
data_type='val'
src_file='/home/hogan/data1/chh/clean_exp/code_classification/data_list/val_3.txt'
dst_dir='/home/hogan/data1/chh/clean_exp/code_classification/data3'
with open(src_file, "r", encoding="utf-8") as f:
    info = f.readlines()
for img_info in info:
    img_path, label = img_info.strip().split('\t')
    dst_path=os.path.join(dst_dir,data_type)
    dst_path=os.path.join(dst_path,label)
    shutil.copy(img_path,dst_path)
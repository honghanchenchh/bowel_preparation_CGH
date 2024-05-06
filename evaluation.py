import os

from sklearn.metrics import classification_report,accuracy_score,recall_score
y_true =[]
with open('4alllist_shuffle.txt','r') as f:
    ret=f.readlines()
    for itm in ret:
        itm=itm.rstrip('\n')
        a,b=itm.split()
        y_true.append(int(b))
y_pred =[]
with open('res/4resalex.txt','r') as f:
    ret=f.readlines()
    for itm in ret:
        itm=itm.rstrip('\n')
        a,b=itm.split()
        y_pred.append(int(b))
target_names = ['L I', 'L II', 'L III', 'L IV']
print(classification_report(y_true, y_pred, target_names=target_names))

'''
file_list=os.listdir(r'D:\整理\蒙晶标注4级\Dahshan_combine\IV')
Flst=0
Mlst=0
for itm in file_list:
    sex_c=itm[5:6]
    if sex_c=='F':
        Flst+=1
    elif sex_c=='M':
        Mlst+=1

print(Flst,Mlst,len(file_list)-Flst-Mlst)
'''
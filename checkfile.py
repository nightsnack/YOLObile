import os
import shutil

imagedir = 'data/coco/images/train2017/'
labeldir = 'data/coco/labels/train2017'
newdir = 'data/coco/empty/train2017'
labellist = os.listdir(labeldir)
emptyfile = 0
for image in os.listdir(imagedir):
    label = image.replace("jpg","txt")
    if label not in labellist:
        print("Label {} not exits".format(label))
        emptyfile = emptyfile+1
        shutil.move(imagedir+image, newdir)
print(emptyfile)

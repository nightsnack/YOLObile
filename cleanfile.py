# f = open('data/coco/val2017.txt', 'r')
# f1 = open('data/coco/val2017.shapes', 'r')
import os

imagedir = 'data/coco/images/train2017/'
newdir = 'data/coco/empty/train2017'
nonemptyfile = 0
imagelist = os.listdir(newdir)

with open('data/coco/train2017.txt', 'r') as r:
    lines=r.readlines()
with open('data/coco/train2017.shapes', 'r') as r:
    sline=r.readlines()
with open('data/coco/train2020.txt','w') as w:
    with open('data/coco/train2020.shapes', 'w') as ws:
        for i in range(len(lines)):
            imagename = lines[i][-17:-1]
            if imagename not in imagelist:
                w.write(lines[i])
                ws.write(sline[i])
                nonemptyfile = nonemptyfile+1

print(nonemptyfile)

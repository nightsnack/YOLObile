python train.py --img-size 320 --batch-size 64 --device 0,1,2,3 --epoch 25 --admm-file darknet_admm --cfg cfg/csdarknet53s-panet-spp.cfg --weights weights/best563.pt --data data/coco2014.data
echo finish admm process
python train.py --img-size 320 --batch-size 64 --device 0,1,2,3 --epoch 280 --admm-file darknet_retrain --cfg cfg/csdarknet53s-panet-spp.cfg --weights weights/best563.pt --data data/coco2014.data --multi-scale
echo finish retrain
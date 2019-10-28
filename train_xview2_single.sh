python train.py \
 --backbone resnet \
 --lr 0.007 \
 --workers 4 \
 --epochs 20 \
 --batch-size 3 \
 --loss-type ce \
 --gpu-ids 0 \
 --checkname deeplab-resnet \
 --eval-interval 1 \
 --dataset xview2

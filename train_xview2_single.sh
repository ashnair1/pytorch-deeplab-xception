python train.py \
 --backbone drn \
 --lr 0.1 \
 --workers 4 \
 --epochs 50 \
 --batch-size 32 \
 --loss-type ce \
 --base-size 1024 \
 --crop-size 256 \
 --gpu-ids 0,1,2,3,4,5,6,7 \
 --checkname deeplab-drn \
 --ft \
 --use-balanced-weights \
 --eval-interval 1 \
 --dataset xview2_single

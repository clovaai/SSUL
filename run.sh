DATA_ROOT=your_dataset_root_path
DATASET=voc
TASK=15-1
EPOCH=50
BATCH=16
LOSS=bce_loss
LR=0.01
THRESH=0.7
MEMORY=0

python main.py --data_root ${DATA_ROOT} --model deeplabv3_resnet101 --gpu_id 0,1 --crop_val --lr ${LR} \
    --batch_size ${BATCH} --train_epoch ${EPOCH}  --loss_type ${LOSS} \
    --dataset ${DATASET} --task ${TASK} --overlap --lr_policy poly \
    --pseudo --pseudo_thresh ${THRESH} --freeze  --bn_freeze  \
    --unknown --w_transfer --amp --mem_size ${MEMORY}

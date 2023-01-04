export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=$PWD:$PYTHONPATH 
nohup python train.py --dataset vkitti2 --gpus 8 --lr 0.00025 --name train_vkitti2 > train.log 2>&1 &
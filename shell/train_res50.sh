conda activate homefun
cd homefun/zhf/FasterDETR
CUDA_VISIBLE_DEVICES=1,5 nohup python -m torch.distributed.launch --nproc_per_node=2 main.py >weights/fasterdetr_log.out 2>&1 &

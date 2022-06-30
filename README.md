## 文件结构：
```
  ├── model: SAM-DETR模型搭建部分
  ├── utils: Dataset train_one_epoch evaluate transformer等相关程序
  ├── weights: 保存训练权重，结果等文件
  ├── shell：训练sh文件
  ├── mian.py: 开始训练的程序
  ├── predict.py：预测单张图片，并且输出推理单张图片的时间
  ├── validation.py：获得各个类别的map等参数
  ├── voc_xml_transform_txt.py：把voc数据集的xml文件转成训练所需要的txt文件并保存到指定位置
```
## 训练
* 确保提前准备好数据集
* 使用`voc_xml_transform_txt.py`转成训练所需的txt文件格式（例如:[train.txt](weights/train.txt)）
* 若要使用多GPU训练，使用`python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py`指令,`nproc_per_node`参数为使用GPU数量
* 如果想指定使用哪些GPU设备可在指令前加上`CUDA_VISIBLE_DEVICES=1,5`(例如我只要使用设备中的第2块和第6块GPU设备)，详情见[train.sh](shell/train_res50.sh)
* 训练过程中的lr与loss：![lr and loss](weights/loss_20220612222954/loss_and_lr.png)
* 训练过程中的map(IoU=0.5)：![map(IoU=0.5)](weights/loss_20220612222954/mAP.png)
* [训练过程所有输出](weights/fasterdetr_log.out)
## 预测
* 需要修改`--class_path`指向记录类别的txt文件（例如[voc_classes.txt](weights/voc_classes.txt)）
* 需要修改`--pic_path`指向需要预测的图片地址（例如[predict.jpg](2009_003351.jpg)）
* 需要修改`--weights_path`指向训练保存的权重文件（例如[FasterDETR.pth](weights/loss_20220612222954/resnet50_FasterDETR_bestMap.pth)）
* `python predict.py`
* 预测结果：![predict for SAM-DETR](test.jpg)
## 获得各个类别的map等参数
* 需要修改`--class_path`指向记录类别的txt文件（例如[voc_classes.txt](weights/voc_classes.txt)）
* 需要修改`--val`指向验证集/测试集txt文件（例如[val.txt](weights/val.txt)）
* 需要修改`--weights_path`指向训练保存的权重文件（例如[FasterDETR.pth](weights/loss_20220612222954/resnet50_FasterDETR_bestMap.pth)）
* `python validation.py`
* 验证结果：[record_mAP.txt](weights/loss_20220612222954/record_mAP.txt)

# 效果
以双GPU为例，例程在multi_gpu中占用量为11GB/卡，训练时长为4h/epoch，在zero2中占用量为8GB/卡，训练时长为8h/epoch，其中内存占用量比理论结果（9/16）略大，可能是训练时每张卡还要加载训练用的东西（约1.5GB）

# 如何使用
以 Kaggle 为例：
1. 下载仓库
```python
!git clone https://github.com/Kamichanw/SeekDeeper.git
```
2. 切换工作目录
```python
!cd SeekDeeper
```
3. 安装依赖 && 删除冲突库
```python
!pip install -r /kaggle/working/SeekDeeper/accelerateGPT/requirements.txt
!pip uninstall -y nvtx
```
4. 运行训练脚本
```python
!accelerate launch \
--config_file /kaggle/working/SeekDeeper/accelerateGPT/config_files/zero2.yaml \
/kaggle/working/SeekDeeper/accelerateGPT/accelerate_pretrain.py \
--use_tensorboard
```
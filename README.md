# IVQA 算法

## 环境安装
* python环境: python3.7。
* 安装python包：`pip install -r requirement.txt`。

## 文件结构
1. [checkpoints](./checkpoints) 存放训练好的模型
2. [frames](./frames) 存放数据
3. [results](./results) 存放推理结果


## 运行

### 推理
```
python inference_compact.py --model ./checkpoints/NDG.model --folder ./frames/ --imageformat png --data_type image_folders
```


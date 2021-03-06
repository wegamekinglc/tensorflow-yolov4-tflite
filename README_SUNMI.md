# tensorflow-yolov4-tflite

## 代码库

`http://120.26.148.12/gitlab-ci-group/tensorflow-yolov4-tflite`

## 环境

* 开发机: `172.16.0.174`
* 代码地址: `/data/dev/cheng/remote/tf2-yolov4`
* 数据地址: 
    * 训练数据: `/data/dev/zoe/heads_data/v0.1.0`
    * 测试数据: `/data/dev/cheng/heads_data`
* python环境：`conda activate handsonml2`

## 训练

```bash
(handsonml2) $ python train_sunmi.py
```

脚本会在`/data/dev/cheng/remote/tf2-yolov4/checkpoints/v4`目录下，每10个epoch保存模型的权重：

```bash
$ ls ./checkpoints/v4
yolov4_100.data-00000-of-00001  yolov4_170.index                yolov4_250.data-00000-of-00001  yolov4_320.index                yolov4_40.data-00000-of-00001   yolov4_480.index                yolov4_560.data-00000-of-00001  yolov4_630.index
yolov4_100.index                yolov4_180.data-00000-of-00001  yolov4_250.index                yolov4_330.data-00000-of-00001  yolov4_40.index                 yolov4_490.data-00000-of-00001  yolov4_560.index                yolov4_640.data-00000-of-00001
yolov4_10.data-00000-of-00001   yolov4_180.index                yolov4_260.data-00000-of-00001  yolov4_330.index                yolov4_410.data-00000-of-00001  yolov4_490.index                yolov4_570.data-00000-of-00001  yolov4_640.index
yolov4_10.index                 yolov4_190.data-00000-of-00001  yolov4_260.index                yolov4_340.data-00000-of-00001  yolov4_410.index                yolov4_500.data-00000-of-00001  yolov4_570.index                yolov4_650.data-00000-of-00001
yolov4_110.data-00000-of-00001  yolov4_190.index                yolov4_270.data-00000-of-00001  yolov4_340.index                yolov4_420.data-00000-of-00001  yolov4_500.index                yolov4_580.data-00000-of-00001  yolov4_650.index
```

## 保存模型

### base model

使用一下命令，在`./models/`下生产一个基于darknet pretrained weights的基础模型：

```bash
(handsonml2) $ python save_model.py
```

生成的结果为：

```bash
$ ls -l models/yolov4-416/
total 10892
drwxr-xr-x 2 sail sail     4096 7月  13 14:03 assets
-rw-rw-r-- 1 sail sail 11141239 8月   7 15:35 saved_model.pb
drwxr-xr-x 2 sail sail     4096 8月   7 15:35 variables
```

### 更新权重

假设我们在训练模型阶段，得到如下的权重文件：

```
-rw-rw-r-- 1 sail sail 257783985 7月  16 09:03 ./checkpoints/v4/yolov4_630.data-00000-of-00001
-rw-rw-r-- 1 sail sail     34706 7月  16 09:03 ./checkpoints/v4/yolov4_630.index
```

我们将这些权重文件拷贝至base model的`variables`文件夹，并改为对应的名字：
```bash
-rw-rw-r-- 1 sail sail 258013298 8月   7 15:35 variables.data-00000-of-00001
-rw-rw-r-- 1 sail sail     34706 8月   7 15:35 variables.index
```

## 测试

直接运行：

```bash
(handsonml2) $ python detect_sunmi.py
```

运行结束之后，可以：

1. 图片文件：`./data/results`目录下看到结果。
2. 检测结果信息：
    * `/data/dev/cheng/heads_data/20200507_mozi.json`
    * `/data/dev/cheng/heads_data/20200521_coolrat.json`

## 模型评估

模型评估我们使用`model-eval: http://120.26.148.12/gitlab-ci-group/model-eval`

评估脚本在`examples/3. head_model_eval.py`

```bash
$ python 3.\ head_model_eval.py
TEST - MOZI
             ap        ar
IOU
0.50  0.959639  0.967366
0.55  0.959639  0.967366
0.60  0.959593    0.9662
0.65   0.95951  0.965035
0.70  0.959464  0.963869
0.75  0.959464  0.963869
0.80   0.94932  0.956876
0.85   0.93767  0.945221
0.90  0.922752  0.931235
0.95   0.74064  0.817016
Avg.  0.930769  0.944406


TEST - COOLRAT
             ap        ar
IOU
0.50  0.858637  0.867457
0.55  0.858637  0.867457
0.60  0.858637  0.867457
0.65  0.858507  0.866379
0.70  0.857649  0.864224
0.75  0.856919  0.860991
0.80  0.844243  0.850216
0.85  0.805951  0.823276
0.90  0.737433  0.773707
0.95  0.498322  0.621228
Avg.  0.803493  0.826239
```


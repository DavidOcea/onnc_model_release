# 简介
detection coco onnx 模型推理脚本

- 目录结构

```
20220705_detection_onnx_model_release
├── inference.py
├── utils.py
├── test_imgs
│   └── 000000000139.jpg
└── onnx_model
    └── detection_coco_model.onnx
```
* inference.py 推理脚本，可以推理单张图片
* utils.py 工具包
* detection_coco_model.onnx 待推理模型 模型输入：[608,608]; 均值：[0.406, 0.456, 0.485] 方差：[0.225, 0.224, 0.229]

```

## 图片demo测试

```shell
cd 20220705_detection_onnx_model_release
python inference.py --imgpath test_imgs/000000000139.jpg \
                    --input_shape (608, 608) \
                    --model_path onnx_model/detection_coco_model.onnx
```


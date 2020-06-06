# NTUST--MaskFaceDetection
To detect human beings wearing mask or not. 

## Result
| no mask | wearing mask | blablabl|
| -------- | -------- | --- |
|  ![](https://i.imgur.com/MwvkHV5.jpg) | ![](https://i.imgur.com/qRPOp6o.jpg)| ![](https://i.imgur.com/tRR7TnD.jpg)|

## What did I do?
Integrate [Yoloface project](https://github.com/sthanhng/yoloface) and my mask classification model that I train by myself.

[Mask Face Data source](https://www.kaggle.com/andrewmvd/face-mask-detection)

## Requirements
```
numpy
tensorflow>=1.12.1
opencv
keras
matplotlib
pillow
```

## How to run?
First thing you have to do is to create a directory "./mask_detection/model-weights", and then download models to the directory from [here](https://drive.google.com/file/d/1mRS5c5K-qGSzGc_Ex-3F-oH1GJLyv_CJ/view?usp=sharing).
Then:
```
$ cd mask_detection
$ python .\yoloface_gpu.py --video  stream
```

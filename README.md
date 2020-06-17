# NTUST--MaskFaceDetection
To detect human beings wearing mask or not. 

## Result
| no mask | wearing mask | blablabl|
| -------- | -------- | --- |
|  ![](https://i.imgur.com/MwvkHV5.jpg) | ![](https://i.imgur.com/qRPOp6o.jpg)| ![](https://i.imgur.com/tRR7TnD.jpg)|

## What did I do?
Two way approach:
- Integrate Openvino pretrained IR model with my mask classifier (Faster, 1 frame per 0.03~0.1 seconds with i7 cpu)
- Integrate [Yoloface project](https://github.com/sthanhng/yoloface) and my mask classifier. (Slower, 1 frame per 0.7~0.8 second with i7 cpu)

[Mask Classifier training data source](https://www.kaggle.com/andrewmvd/face-mask-detection)


## For openvino project
### Requirements
Install [Openvino](https://docs.openvinotoolkit.org/latest/index.html) first.
```
numpy
tensorflow>=1.12.1
opencv
keras
pillow
```
### How to run?
```
$ cd openvino_ir_model
$ {your path to openvino directory}\bin\setupvars.bat
$ python .\mask_detector.py
```
| argument | default | description |
| -------- | -------- | -------- |
| --mode     | 'webcam'     | you can choose which mode('webcam', 'image', 'video') you want to try   |
| -f, --face_threshold     |  0.5    |  IR model will give each face a confidence, this threshold can restrict the face to display |
| --input_file     | ''     | In image or video model, you have to set an input_file path     |
| -save_path | ''  | Output file path of your demo  |



## For yoloface project
### Requirements
```
numpy
tensorflow>=1.12.1
opencv
keras
matplotlib
pillow
```
### How to run?
First thing you have to do is to create a directory "./yoloface_model/model-weights", and then download models to the directory from [here](https://drive.google.com/file/d/1mRS5c5K-qGSzGc_Ex-3F-oH1GJLyv_CJ/view?usp=sharing).
Then:
```
$ cd yoloface_model
$ python .\yoloface_gpu.py --video  stream
```

# Masked Face Detection(openvino)

## Objective
Collect the people’s’ information while they get into/out a place e.g. convenience store, school, etc. And then we can do some analysis.
People’s’ information contains
- Wearing mask or not
- Age
- Gender
- Timestamp

![](https://i.imgur.com/7olh273.png)
# Requirements
Install [Openvino](https://docs.openvinotoolkit.org/latest/index.html) first.
```
numpy
tensorflow>=1.12.1
opencv
keras
dlib
pandas
pyqt5
```

# How to run?
```
$ cd mask_detection
$ {your path to openvino directory}\bin\setupvars.bat
$ python .\main.py
```

| argument | default | description |
| -------- | -------- | -------- |
| face_threshold     |  0.5    |  IR face-detection model will give each face a confidence, this threshold can restrict the face to display |
| input_file     | ''     | test_video path  (\*.mp4, \*.avi), if it's not specified, we will read webcam.|
| save_video | ''  | save_video path(\*.mp4, \*.avi)|
| save_data | ''  | save_collected_data path(\*.csv) |

## What did I do?
1. Train a MaskFaceClassifier
    - In *mask_classifier_model_training*
3. Use openvino pretrained model to detect face and classify face age and gender
    - [face-detection-adas-0001](https://docs.openvinotoolkit.org/2019_R1/_face_detection_adas_0001_description_face_detection_adas_0001.html)
    - [age-gender-recognition-retail-0013](https://docs.openvinotoolkit.org/latest/_models_intel_age_gender_recognition_retail_0013_description_age_gender_recognition_retail_0013.html)
4. Add Tracker to speed up
    - [Pyimagesearch-CentroidTracker](https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/)

4. Add crossline and build a gui (while a person crossing the line, right side will pop up the classify result)
5. Add DataCollector to collect data from classifier


## For classifier training
[Data source](https://www.kaggle.com/andrewmvd/face-mask-detection)

![](https://i.imgur.com/X1MBzXi.png)

![](https://i.imgur.com/N1ifxZI.png)

![](https://i.imgur.com/5r6ftLp.png)
![](https://i.imgur.com/67J3Dea.png)

## For tracker
Pros: 
- Have higher speed. 
- Detection is more computation expensive than tracker. (And we only have cpu.)

Cons:
- Have lower accuracy.
- Tracker track the face by correlation of image

![](https://i.imgur.com/xzh6zbp.png)
![](https://i.imgur.com/GhooQER.png)

![](https://i.imgur.com/x8USO79.png)
![](https://i.imgur.com/ZIj8MhR.png)

![](https://i.imgur.com/lJPNJrm.png)
![](https://i.imgur.com/CBGeMtJ.png)

## Data collected
![](https://i.imgur.com/e4vibjz.png)

## What can we do after collect these data?
![](https://i.imgur.com/61qgRSB.png)



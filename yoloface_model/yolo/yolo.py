# *******************************************************************
#
# Author : Thanh Nguyen, 2018
# Email  : sthanhng@gmail.com
# Github : https://github.com/sthanhng
#
# Face detection using the YOLOv3 algorithm
#
# Description : yolo.py
# Contains methods of YOLO
#
# *******************************************************************

import os
import colorsys
import numpy as np
import cv2

from yolo.model import eval

from keras import backend as K
from keras.models import load_model
from timeit import default_timer as timer
from PIL import ImageDraw, Image

# me
from mask_model import model


class DetectMachine(object):
    def __init__(self, args):
        self.args = args
        self.model_path = args.model
        self.classes_path = args.classes
        self.anchors_path = args.anchors
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self._generate()
        self.model_image_size = args.img_size
        
        # me
        self.mask_model_face_size = args.face_size
        self.mask_model = model.get_model((args.face_size[0], args.face_size[1], 3))
        self.mask_model.load_weights(args.mask_model)
        
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def _generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith(
            '.h5'), 'Keras model or weights must be a .h5 file'

        # load model, or construct model and load weights
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            # make sure model, anchors and classes match
            self.yolo_model.load_weights(self.model_path)
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (
                           num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'
        print(
            '*** {} model, anchors, and classes loaded.'.format(model_path))

        # generate colors for drawing bounding boxes
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        # shuffle colors to decorrelate adjacent classes.
        np.random.seed(102)
        np.random.shuffle(self.colors)
        np.random.seed(None)

        # generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        boxes, scores, classes = eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names),
                                           self.input_image_shape,
                                           score_threshold=self.args.score,
                                           iou_threshold=self.args.iou)
        return boxes, scores, classes

    def run(self, image):
        
        # detect face
        if self.model_image_size != (None, None):
            assert self.model_image_size[
                       0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[
                       1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(
                reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        # add batch dimension
        
        image_data = np.expand_dims(image_data, 0)
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
            
        # me
        out_boxes = np.array([[max(0, np.floor(box[0] + 0.5).astype('int32')),
                                max(0, np.floor(box[1] + 0.5).astype('int32')),
                                min(image.size[1], np.floor(box[2] + 0.5).astype('int32')),
                                min(image.size[0], np.floor(box[3] + 0.5).astype('int32'))
                                ] for box in out_boxes])
        
        print('*** Found {} face(s) for this image'.format(len(out_boxes)))
        
        
        # classify faces
        im = np.array(image) 
        if len(out_boxes) > 0:
            images = np.array([cv2.resize(im[box[0]:box[2], box[1]:box[3]], self.mask_model_face_size) for box in out_boxes])
            preds = self.mask_model.predict(images).flatten()
        else:
            preds = np.array([])
        
        
        # draw image
        thickness = (image.size[0] + image.size[1]) // 300
        for i, (box, pred) in reversed(list(enumerate(zip(out_boxes, preds)))):
            box = out_boxes[i]
            text = 'wearing mask probability: {}'.format(pred)
            draw = ImageDraw.Draw(image)

            top, left, bottom, right = box
            print(text, (left, top), (right, bottom))
            
            color = (255, 255, 0) if pred > 0.5 else (0, 255, 0)
            # draw bounding box
            for thk in range(thickness):
                draw.rectangle(
                    [left + thk, top + thk, right - thk, bottom - thk],
                    outline = color)
                
            # put text
            heigth = 20
            if top - heigth >= 0:
                display_text = "mask" if pred > 0.5 else "no mask"
                draw.rectangle([left, top - heigth, right, top],
                               fill=color)
                draw.text((left + thickness + 1, top - heigth), display_text, fill=(0,0,0))
                
            del draw
        return image, out_boxes
        

    def close_session(self):
        self.sess.close()


def letterbox_image(image, size):
    '''Resize image with unchanged aspect ratio using padding'''

    img_width, img_height = image.size
    w, h = size
    scale = min(w / img_width, h / img_height)
    nw = int(img_width * scale)
    nh = int(img_height * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def detect_img(detectMachine):
    while True:
        img = input('*** Input image filename: ')
        try:
            image = Image.open(img)
        except:
            if img == 'q' or img == 'Q':
                break
            else:
                print('*** Open Error! Try again!')
                continue
        else:
            image, _ = detectMachine.run(image)
            image.show()
    detectMachine.close_session()


def detect_video(detectMachine, video_path=None, output=None):
    
    if video_path == 'stream':
        vid = cv2.VideoCapture(0)
    else:
        vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")

    # the video format and fps
    # video_fourcc = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fourcc = cv2.VideoWriter_fourcc('M', 'G', 'P', 'G')
    video_fps = vid.get(cv2.CAP_PROP_FPS)

    # the size of the frames to write
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    
    
    isOutput = True if output != "" else False
    if isOutput:
        output_fn = 'output_video.avi'
        out = cv2.VideoWriter(os.path.join(output, output_fn), video_fourcc, video_fps, video_size)

    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    ###################################
    while True:
        ret, frame = vid.read()
        if ret:
            image = Image.fromarray(frame)
            
            start_time = timer()
            image, out_boxes = detectMachine.run(image)
            end_time = timer()
            print('*** Processing time: {:.2f}ms'.format((end_time -
                                                          start_time) * 1000))
            result = np.asarray(image)
            
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = curr_fps
                curr_fps = 0

            # initialize the set of information we'll displaying on the frame
            info = [
                ('FPS', '{}'.format(fps)),
                ('Faces detected', '{}'.format(len(out_boxes)))
            ]
            cv2.rectangle(result, (5, 5), (120, 50), (0, 0, 0), cv2.FILLED)

            for (i, (txt, val)) in enumerate(info):
                text = '{}: {}'.format(txt, val)
                cv2.putText(result, text, (10, (i * 20) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (10, 175, 0), 1)

            cv2.namedWindow("face", cv2.WINDOW_NORMAL)
            cv2.imshow("face", result)
            if isOutput:
                out.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    vid.release()
    out.release()
    cv2.destroyAllWindows()
    # close the session
    model.close_session()
    
    
    
    
    
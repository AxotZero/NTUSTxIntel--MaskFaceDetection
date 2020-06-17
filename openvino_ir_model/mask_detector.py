# -*- coding: utf-8 -*-
import cv2
import numpy as np
import datetime
import argparse

from mask_classifier_model import model_structure


face_threshold = 0.5

def process_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', type=str, choices=['image', 'video', 'webcam'], default='webcam')
	
	parser.add_argument('-f', '--face_threshold', type=float, default=0.5, 
						help='the face detection threshold')
	
	parser.add_argument('--input_file', type=str, default='samples/subway.mp4',
					 help='test file path')
	
	parser.add_argument('--save_path', type=str, default='',
                        help='result path')
	
	args = parser.parse_args()
	return args


def load_face_detection_model():
	FP = 32
	targetId = 0
	weights = 'face_detection_model/FP{}/face-detection-adas-0001.bin'.format(FP)
	config = 'face_detection_model/FP{}/face-detection-adas-0001.xml'.format(FP)
	framework = 'DLDT'
	face_detection_model = cv2.dnn.readNet(weights, config, framework)
	face_detection_model.setPreferableTarget(targetId=targetId)
	
	print('load_face_detection_model done.')
	return face_detection_model


def load_mask_classifier_model():
	MODEL_PATH = 'mask_classifier_model/mask.h5'
	mask_model = model_structure.get_model((128,128,3))
	mask_model.load_weights(MODEL_PATH)
	
	print('load_mask_classifier_model done.')
	return mask_model
		

def crop_img(frame, start_x, start_y, end_x, end_y):
	if start_x < 0: start_x = 0
	if start_y < 0: start_y = 0
	if end_x > frame.shape[1]: end_x = frame.shape[1]
	if end_y > frame.shape[0]: end_y = frame.shape[0]
	
	INPUT_SIZE = (128, 128)
	face_img = frame[start_y:end_y, start_x:end_x, :]
	face_img = cv2.resize(face_img, INPUT_SIZE)
	return face_img


def draw_bbox(frame, start_x, start_y, end_x, end_y, have_mask):
    color = (0, 255, 0) if have_mask else (255, 0, 0)
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)

def draw_info(frame, face_detected, mask_detected):
	cv2.rectangle(frame, (5, 5), (200, 60), (0, 0, 0), cv2.FILLED)
	
	txts = [
		'Face detected: {}'.format(face_detected), 
		'Mask detected: {}'.format(mask_detected), 
	]
	for i, txt in enumerate(txts):
		cv2.putText(frame, txt, (10, (i+1)*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 175, 0), 1)
		
	
	
def detect_image(image, face_detection_model, mask_clissifier_model):
	h, w, c = image.shape
	blob = cv2.dnn.blobFromImage(image, size=(w, h), crop=False)
	face_detection_model.setInput(blob)
	
	bboxes = face_detection_model.forward()
	bboxes = (bboxes[0, 0, bboxes[0, 0, :, 2] > face_threshold][:, 3:] * np.array([w, h, w, h])).astype(int)
	
	
	print('detected %d faces' % len(bboxes), 'at', bboxes)
	
	if(len(bboxes) == 0):
		draw_info(image, 0, 0)
		
	else:
		faces = np.array([crop_img(image, bbox[0], bbox[1], bbox[2], bbox[3]) for bbox in bboxes]) / 255.
		mask_result = mask_clissifier_model.predict(faces).flatten()
		mask_result = [1 if result > 0.5 else 0 for result in mask_result]
		for i in range(len(bboxes)):
			draw_bbox(image, bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3], mask_result[i])
	
		draw_info(image, len(bboxes), sum(mask_result))
	

def run_image_detection(input, save_path):
	
	face_detection_model = load_face_detection_model()
	mask_clissifier_model = load_mask_classifier_model()
	
	frame = cv2.imread(input)
	start_time = datetime.datetime.now()
	
	detect_image(frame, face_detection_model, mask_clissifier_model)
	cv2.imshow('frame',frame)
	
	# end read img
	end_time = datetime.datetime.now()
	total_time = (end_time - start_time).seconds + (end_time - start_time).microseconds * 1E-6
	print("spend: {}s".format(total_time))
	
	if save_path != '':
		cv2.imwrite(frame)
		
	if cv2.waitKey(10000) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
	
	
def run_video_detection(input, save_path):
	
	if input == 'webcam':
		vid = cv2.VideoCapture(0)
	else:
		vid = cv2.VideoCapture(input)
	
	if not vid.isOpened():
		raise IOError("Couldn't open video")
	
	if save_path != '':
		video_fourcc = cv2.VideoWriter_fourcc('M', 'G', 'P', 'G')
		video_fps = vid.get(cv2.CAP_PROP_FPS)
		video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
		save_video = cv2.VideoWriter(save_path, video_fourcc, video_fps, video_size)
		
		
	face_detection_model = load_face_detection_model()
	mask_clissifier_model = load_mask_classifier_model()
	
	while True:
		start_time = datetime.datetime.now()
		
		ret, frame = vid.read()
		if not ret:
			break
		
		detect_image(frame, face_detection_model, mask_clissifier_model)
		cv2.imshow('frame',frame)
		
		if save_path != '':
			save_video.write(frame)
			
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		
		# end read img
		end_time = datetime.datetime.now()
		total_time = (end_time - start_time).seconds + (end_time - start_time).microseconds * 1E-6
		print("spend: {}s".format(total_time))
		
	
	vid.release()
	cv2.destroyAllWindows()
	if save_path != '': 
		save_video.release()

		
def main():
	global face_threshold
	
	args = process_args()
	face_threshold = args.face_threshold
	
	if args.mode == 'webcam':
		run_video_detection('webcam', args.save_path)
	elif args.mode == 'video':
		run_video_detection(args.input_file, args.save_path)
	else:
		run_image_detection(args.input_file, args.save_path)
	
		
if __name__ == '__main__':
	main()

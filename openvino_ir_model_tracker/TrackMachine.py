

from pyimagesearch.ObjectTracker import ObjectTracker
from pyimagesearch.TrackableObject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

import datetime
from mask_classifier_model import mask_model
GENDERS_FOR_OPENVINO = ['Female', 'Male']

def cross(p1, p2, p3):#跨立实验
    x1=p2[0]-p1[0]
    y1=p2[1]-p1[1]
    x2=p3[0]-p1[0]
    y2=p3[1]-p1[1]
    return x1*y2-x2*y1  

def crop_img(frame, bbox):
	start_x, start_y, end_x, end_y = bbox
	if start_x < 0: start_x = 0
	if start_y < 0: start_y = 0
	if end_x > frame.shape[1]: end_x = frame.shape[1]
	if end_y > frame.shape[0]: end_y = frame.shape[0]
	
	face_img = frame[start_y:end_y, start_x:end_x, :]

	return face_img

class FaceDetector:

	def __init__(self, confidence_threshold=0.8):
		self.confidence_threshold = confidence_threshold
		self.model = self.load_model()

	def load_model(self, model_path='face_detection_model/FP32/face-detection-adas-0001'):
		targetId = 0
		weights = model_path+'.bin'
		config = model_path+'.xml'
		framework = 'DLDT'

		try:
			face_detection_model = cv2.dnn.readNet(weights, config, framework)
			face_detection_model.setPreferableTarget(targetId=targetId)
		except:
			print('Unable to read face detector model')
			raise IOError
		
		print('load_face_detection_model done.')
		return face_detection_model

	def detect(self, cv2_image):
		h, w, c = cv2_image.shape
		blob = cv2.dnn.blobFromImage(cv2_image, size=(w, h), crop=False)
		self.model.setInput(blob)
		bboxes = self.model.forward()
		bboxes = bboxes[0, 0, bboxes[0, 0, :, 2] > self.confidence_threshold][:, 3:]
		bboxes = (bboxes * np.array([w, h, w, h])).astype(int)
		return bboxes

class FaceClassifier:
	def __init__(self):
		self.mask_classifier = self.load_mask_classifier_model()
		self.age_gender_model = self.load_age_gender_model()

	def load_mask_classifier_model(self):
		path = 'mask_classifier_model/mask.h5'

		try:
			mask_classifier = mask_model.get_model((128,128,3))
			mask_classifier.load_weights(path)
		except:
			print('Unable to read face detector model')
			raise IOError

		print('load_mask_classifier_model done.')
		return mask_classifier

	def load_age_gender_model(self, model_path='face_age_gender/FP32/age-gender-recognition-retail-0013'):
		targetId = 0
		weights = model_path+'.bin'
		config = model_path+'.xml'
		framework = 'DLDT'

		try:
			face_age_gender_model = cv2.dnn.readNet(weights, config, framework)
			face_age_gender_model.setPreferableTarget(targetId=targetId)
		except:
			print('Unable to read face detector model')
			raise IOError
		
		print('face_age_gender_model done.')
		return face_age_gender_model


	def classify(self, image, bboxes):
		faces = [crop_img(image, bbox) for bbox in bboxes]

		mask_classifier_input = np.array([cv2.resize(face, (128,128)) for face in faces])
		mask_result = self.mask_classifier.predict(mask_classifier_input).flatten()
		mask_result = [1 if result > 0.5 else 0 for result in mask_result]

		ages = []
		genders = []
		for face in faces:
			blob = cv2.dnn.blobFromImage(face, size=(62, 62), ddepth=cv2.CV_8U)
			self.age_gender_model.setInput(blob)
			detections = self.age_gender_model.forwardAndRetrieve(['prob', 'age_conv3'])
			gender = GENDERS_FOR_OPENVINO[detections[0][0][0].argmax()]
			age = int(detections[1][0][0][0][0][0] * 100)

			genders.append(gender)
			ages.append(age)

		return mask_result, genders, ages

class Tracker:

	def __init__(self, maxDisappeared=40):
		self.trackers = [] # list of dlib.correlation_tracker()

		self.centroid_tracker = ObjectTracker(maxDisappeared)
		self.tracking_objects = {} 
		

	def refresh_trackers(self, bboxes, rgb_image):
		self.trackers = []
		for box in bboxes:
			(startX, startY, endX, endY) = box
			tracker = dlib.correlation_tracker()
			rect = dlib.rectangle(startX, startY, endX, endY)
			tracker.start_track(rgb_image, rect)
			self.trackers.append(tracker)

	def update_trackers(self, rgb_image):
		for tracker in self.trackers:
			tracker.update(rgb_image)

	def update_tracking_objects(self):
		bboxes = []
		for tracker in self.trackers:
			pos = tracker.get_position()
			# unpack the position object
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())
			# add the bounding box coordinates to the rectangles list
			bboxes.append((startX, startY, endX, endY))

		objects, deletedIDs = self.centroid_tracker.update(bboxes)
	    # loop over the tracked objects
		for (objectID, value) in objects.items():
			tracking_object = self.tracking_objects.get(objectID, None)

			if tracking_object is None:
				tracking_object = TrackableObject(objectID, value['bbox'])

			else:
				tracking_object.update(value['bbox'])

			self.tracking_objects[objectID] = tracking_object

		for objectID in deletedIDs:
			del self.tracking_objects[objectID]

class Monitor:

	def __init__(self, face_confidence=0.8, detect_frames=8, maxDisappeared=20):
		self.face_detector = FaceDetector(face_confidence)
		self.tracker = Tracker(maxDisappeared)
		self.classifier = FaceClassifier()

		self.cross_line = None
		self.detect_frames = detect_frames
		self.total_frames = 0

	def check_cross_the_line(self, tracking_object):
		if self.cross_line is None:
			return False

		a = tracking_object.old_position
		b = tracking_object.new_position

		c = self.cross_line[0]
		d = self.cross_line[1]

		if min(a[0],b[0]) <= max(c[0],d[0]) and min(c[0],d[0]) <= max(a[0],b[0]) and min(a[1],b[1]) <= max(c[1],d[1])and min(c[1],d[1]) <= max(a[1],b[1]):
			if cross(a, b, c) * cross(a, b, d) <= 0 and cross(c, d, a) * cross(c, d, b) <= 0:
				return True
			
		return False

	def get_cross_line_objectIDs(self, tracking_objects):
		to_classified_objectIDs = []
		for objectID, tracking_object in tracking_objects.items():
			if tracking_object.classified is False:
				if self.check_cross_the_line(tracking_object):
					tracking_object.classified = True
					to_classified_objectIDs.append(objectID)
		return to_classified_objectIDs

	def run(self, image):
		rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		if self.total_frames % self.detect_frames == 0:
		# if self.total_frames % 1 == 0:
			bboxes = self.face_detector.detect(image)
			self.tracker.refresh_trackers(bboxes, rgb)
		else:
			self.tracker.update_trackers(rgb)
		self.tracker.update_tracking_objects()

		tracking_objects = self.tracker.tracking_objects
		to_classified_objectIDs = self.get_cross_line_objectIDs(tracking_objects)

		if len(to_classified_objectIDs) > 0:
			bboxes = [tracking_objects[objectID].bbox for objectID in to_classified_objectIDs]
			mask_result, genders, ages = self.classifier.classify(image, bboxes)
			
			for i, objectID in enumerate(to_classified_objectIDs):
				tracking_objects[objectID].mask = mask_result[i]
				tracking_objects[objectID].gender = genders[i]
				tracking_objects[objectID].age = ages[i]
				print('ID:', objectID)
				print('masks:', mask_result)
				print('genders:', genders[i])
				print('ages:', ages[i])
				print()

		self.draw(image, tracking_objects)
		self.total_frames += 1

	def draw(self, image, tracking_objects):
		for objectID, tracking_object in tracking_objects.items():

			text = "ID {}".format(objectID)
			centroid = tracking_object.new_position
			bbox = tracking_object.bbox

			cv2.putText(image, text, (centroid[0] - 10, centroid[1] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			cv2.circle(image, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

			if not tracking_object.classified:
				cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2)
			else:
				cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
				cv2.rectangle(image, (bbox[0]-2, bbox[1]-35), (bbox[2]+2, bbox[1]), (0, 255, 0), cv2.FILLED)

				cv2.putText(image, 'Mask' if tracking_object.mask else 'No Mask', (bbox[0]+5, bbox[1]-20),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
				cv2.putText(image, '%s, %d' %(tracking_object.gender, tracking_object.age) , (bbox[0]+5, bbox[1]-5),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)


if __name__ == "__main__":

	# vid = cv2.VideoCapture(0)
	vid = cv2.VideoCapture('test_data/pe.mp4')
	if not vid.isOpened():
		raise IOError("Couldn't open video or webcam")

	monitor = Monitor()
	cross_line = ((300, 0), (300, 500))
	monitor.cross_line = cross_line

	start_time = datetime.datetime.now()
	while True:
		ret, frame = vid.read()
		if not ret:
			print('cannot read frame')
			break
		monitor.run(frame)
		cv2.line(frame, cross_line[0], cross_line[1], (0, 255, 255), 2)
		cv2.imshow('frame', frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	end_time = datetime.datetime.now()
	total_time = (end_time - start_time).seconds + (end_time - start_time).microseconds * 1E-6
	print("spend: {}s".format(total_time))

	vid.release()
	cv2.destroyAllWindows()



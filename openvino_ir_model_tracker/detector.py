import numpy as np
import cv2
import dlib
import logging
import datetime
import time

from utils.ObjectTracker import ObjectTracker
from utils.TrackableObject import TrackableObject
from utils.tools import crop_img, cross

from model.mask_classifier_model import mask_model


class FaceDetector:
	def __init__(self, confidence_threshold=0.8):
		self.confidence_threshold = confidence_threshold
		self.model = self.load_model()

	def load_model(self):
		model_path='model/face_detection_model/FP32/face-detection-adas-0001'
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

		self.mask_classifier_input_shape = (62, 62, 3)
		self.age_gender_model_input_shape = (62, 62, 3)

		self.mask_classifier = self.load_mask_classifier_model()
		self.age_gender_model = self.load_age_gender_model()

		self.mask_label = ['No Mask', 'Mask']
		self.gender_label = ['Female', 'Male']

	def load_mask_classifier_model(self):
		path = 'model/mask_classifier_model/mask_classifier.h5'

		try:
			mask_classifier = mask_model.get_model(self.mask_classifier_input_shape)
			mask_classifier.load_weights(path)
		except:
			print('Unable to read face detector model')
			raise IOError

		print('load_mask_classifier_model done.')
		return mask_classifier

	def load_age_gender_model(self):
		model_path='model/face_age_gender/FP32/age-gender-recognition-retail-0013'
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

		mask_classifier_input = np.array([cv2.resize(face, self.mask_classifier_input_shape[:2]) for face in faces])
		mask_result = self.mask_classifier.predict(mask_classifier_input).flatten()
		mask_result = [self.mask_label[result > 0.5] for result in mask_result]

		ages = []
		genders = []
		for face in faces:
			blob = cv2.dnn.blobFromImage(face, size=self.age_gender_model_input_shape[:2], ddepth=cv2.CV_8U)
			self.age_gender_model.setInput(blob)
			detections = self.age_gender_model.forwardAndRetrieve(['prob', 'age_conv3'])

			gender = self.gender_label[detections[0][0][0].argmax()]
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


class Detector:
	def __init__(self, face_confidence=0.5, detect_frames=8, maxDisappeared=40):
		self.face_detector = FaceDetector(face_confidence)
		self.tracker = Tracker(maxDisappeared)
		self.classifier = FaceClassifier()

		self.cross_line = None
		self.detect_frames = detect_frames
		self.total_frames = 0

	def check_cross_the_line(self, tracking_object):
		if self.cross_line is None:
			logging.info('crossline is None')
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
					to_classified_objectIDs.append(objectID)
		return to_classified_objectIDs

	def detect(self, image):
		rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		if self.total_frames % self.detect_frames == 0:
			logging.debug('Call detector ~~~')
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
				tracking_objects[objectID].classified = True
				tracking_objects[objectID].classified_timestamp = int(time.time())
				tracking_objects[objectID].mask = mask_result[i]
				tracking_objects[objectID].gender = genders[i]
				tracking_objects[objectID].age = ages[i]
				
				logging.info('ID: %d' % objectID)
				logging.info('masks: %s' % mask_result)
				logging.info('genders: %s' % genders[i])
				logging.info('ages: %d' % ages[i])
				logging.info('\n')

		self.total_frames += 1

		return tracking_objects, to_classified_objectIDs





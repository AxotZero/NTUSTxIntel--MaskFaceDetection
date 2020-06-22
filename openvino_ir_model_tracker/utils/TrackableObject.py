# -*- coding: utf-8 -*-

class TrackableObject:
	def __init__(self, objectID, bbox):

		self.objectID = objectID
		self.bbox = bbox
		self.old_position = ((bbox[0]+bbox[2])//2, (bbox[1]+bbox[3])//2)
		self.new_position = self.old_position
		
		self.classified = False
		self.classified_timestamp = None
		self.mask = None
		self.age = None
		self.gender = None

	def update(self, bbox):
		self.bbox = bbox
		self.old_position = self.new_position
		self.new_position = ((bbox[0]+bbox[2])//2, (bbox[1]+bbox[3])//2)

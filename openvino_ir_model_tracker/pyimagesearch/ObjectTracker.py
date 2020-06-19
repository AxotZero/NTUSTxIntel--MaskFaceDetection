# -*- coding: utf-8 -*-

# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class ObjectTracker():
	def __init__(self, maxDisappeared=50):
		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()
		self.maxDisappeared = maxDisappeared
        
	def register(self, bbox):
		centroid = ((bbox[0]+bbox[2])//2, (bbox[1]+bbox[3])//2)

		self.objects[self.nextObjectID] = {'centroid':centroid, 'bbox': bbox}
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1

	def update(self, rects):

		deletedIDs = []
		if len(rects) == 0:
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1

				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)
					deletedIDs.append(objectID)

			return self.objects, deletedIDs
		
		inputCentroids = np.zeros((len(rects), 2), dtype="int")

		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY)
			
		if len(self.objects) == 0:
			for i in range(len(inputCentroids)):
				self.register(rects[i])
		
		else:
			objectIDs = list(self.objects.keys())
			objectCentroids = [value['centroid'] for value in self.objects.values()]

			D = dist.cdist(np.array(objectCentroids), inputCentroids)
			rows = D.min(axis=1).argsort()
			cols = D.argmin(axis=1)[rows]

			usedRows = set()
			usedCols = set()

			for (row, col) in zip(rows, cols):
				if row in usedRows or col in usedCols:
					continue

				objectID = objectIDs[row]
				self.objects[objectID] = {'centroid':inputCentroids[col], 'bbox': rects[col]}
				self.disappeared[objectID] = 0

				usedRows.add(row)
				usedCols.add(col)
				
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)
			
			if D.shape[0] >= D.shape[1]:
				for row in unusedRows:
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1

					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)
						deletedIDs.append(objectID)
						
			else:
				for col in unusedCols:
					self.register(rects[col])

		return self.objects, deletedIDs
	
	def deregister(self, objectID):
		del self.objects[objectID]
		del self.disappeared[objectID]

		
		
		
		
		
		
		
	
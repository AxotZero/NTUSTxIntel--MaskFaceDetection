import sys
import cv2
from PyQt5 import QtCore
from PyQt5.QtCore  import pyqtSlot
from PyQt5.QtGui import QImage , QPixmap
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi

import logging
import datetime

from utils.tools import crop_img


class ClassifyResult (QWidget):
	def __init__ (self, parent = None):
		super(ClassifyResult, self).__init__(parent)
		self.textQVBoxLayout = QVBoxLayout()
		self.AgeLabel    = QLabel()
		self.GenderLabel  = QLabel()
		self.FaceImageLabel = QLabel()
		self.MaskLabel = QLabel()

		self.textQVBoxLayout.addWidget(self.AgeLabel)
		self.textQVBoxLayout.addWidget(self.GenderLabel)
		self.textQVBoxLayout.addWidget(self.MaskLabel)

		self.allQHBoxLayout  = QHBoxLayout()
		self.allQHBoxLayout.addWidget(self.FaceImageLabel, 0)
		self.allQHBoxLayout.addLayout(self.textQVBoxLayout, 1)
		self.setLayout(self.allQHBoxLayout)
		# setStyleSheet
		self.AgeLabel.setStyleSheet('''color: rgb(0, 0, 255);''')
		self.GenderLabel.setStyleSheet('''color: rgb(255, 0, 0);''')

		# TODO: 如果with mask 改綠色 之類的
		#self.MaskLabel.setStyleSheet('''color: rgb(255, 0, 0);''')
		self.MaskLabel.setStyleSheet('''color: rgb(0, 255, 0);''')

	def setAge (self, text):
		self.AgeLabel.setStyleSheet('''color: rgb(255, 255, 255);background-color: rgb(103, 124, 138);font-size:15px;padding:5px;''')
		self.AgeLabel.setText(text)

	def setGender(self, text):
		if text == "Female":
			self.GenderLabel.setStyleSheet('''color: rgb(255, 255, 255);background-color: rgb(246, 143, 160);font-size:15px;;padding:5px;''')
		if text == "Male":
			self.GenderLabel.setStyleSheet('''color: rgb(255, 255, 255);background-color: rgb(82, 204, 206);font-size:15px;;padding:5px;''')

		self.GenderLabel.setText(text)

	def setMask(self, text):
		if text == "Mask":
			self.MaskLabel.setStyleSheet('''color: rgb(255, 255, 255);background-color: rgb(149, 212, 122);font-size:15px;;padding:5px;''')
		if text == "No Mask":
			self.MaskLabel.setStyleSheet('''color: rgb(255, 255, 255);background-color: rgb(239, 62, 91);font-size:15px;;padding:5px;''')

		self.MaskLabel.setText(text)

	def setFace(self, img):
		qformat=QImage.Format_RGB888
		img = cv2.resize(img, (48, 48))
		img = QImage(img, img.shape[1], img.shape[0], qformat)
		img = img.rgbSwapped()
		self.FaceImageLabel.setPixmap(QPixmap.fromImage(img))

class Gui(QDialog):
	def __init__(self, detector, collector, args):
		super(Gui,self).__init__()
		loadUi("gui/Gui.ui",self)

		self.exitGUI.clicked.connect(self.exit)
		self.DrawLine.clicked.connect(self.drawCrossLine)
		self.start.clicked.connect(self.run)

		self.args = args
		self.detector = detector
		self.collector = collector
		self.status = 'waiting'

		self.vid = None
		self.crossline = None
		self.initialize()

	def initialize(self):
		args = self.args

		if args.input_file == '':
			vid = cv2.VideoCapture(0)
		else:
			vid = cv2.VideoCapture(args.input_file)

		if not vid.isOpened():
			raise IOError("Couldn't open video or webcam")

		ret, self.firstFrame = vid.read()
		if not ret:
			raise IOError('Cannot read frame')
		
		self.vid = vid
		self.displayImage(self.firstFrame)
		

	@pyqtSlot()
	def run(self):
		self.status = 'running'

		args = self.args
		vid = self.vid
		del self.firstFrame

		self.detector.cross_line = self.getCrossLine()

		if args.save_video != '':
			video_fourcc = cv2.VideoWriter_fourcc('M', 'G', 'P', 'G')
			video_fps = vid.get(cv2.CAP_PROP_FPS)
			video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
			save_video = cv2.VideoWriter(args.save_video, video_fourcc, video_fps, video_size)

		detector = self.detector

		while self.status == 'running':
			start_time = datetime.datetime.now()

			ret, frame = vid.read()
			if not ret:
				print('cannot read frame')
				break

			tracking_objects, classified_objectID = detector.detect(frame)

			for objectID in classified_objectID:
				face = crop_img(frame, tracking_objects[objectID].bbox).copy()
				self.AddClassifyResult(face, tracking_objects[objectID])
				if self.collector is not None:
					self.collector.add(tracking_objects[objectID])

			self.draw(frame, tracking_objects)

			if args.save_video != '':
				save_video.write(frame)

			self.displayImage(frame)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

			end_time = datetime.datetime.now()
			total_time = (end_time - start_time).seconds + (end_time - start_time).microseconds * 1E-6
			logging.debug("spend: {}s".format(total_time))

		vid.release()
		cv2.destroyAllWindows()
		if args.save_video != '':
			save_video.release()

	def getCrossLine(self):
		cross_line = ((int(self.crossLine0x.text()),int(self.crossLine0y.text())),(int(self.crossLine1x.text()),int(self.crossLine1y.text())))
		return cross_line

	def drawCrossLine(self):
		self.crossline = self.getCrossLine()
		copyFrame = self.firstFrame.copy()
		cv2.line(copyFrame, self.crossline[0], self.crossline[1], (0, 255, 255), 2)
		self.displayImage(copyFrame)

	def AddClassifyResult(self, face, tracking_object):
		# 倒序啦幹
		self.listWidget.setSortingEnabled(1)

		myQCustomQWidget = ClassifyResult()
		myQCustomQWidget.setAge("%s years old" % str(tracking_object.age))
		myQCustomQWidget.setGender(tracking_object.gender)
		myQCustomQWidget.setMask(tracking_object.mask)
		myQCustomQWidget.setFace(face)

		myQListWidgetItem = QListWidgetItem(self.listWidget)
		# Set size hint 不加的話 myQListWidgetItem size 會為0
		myQListWidgetItem.setSizeHint(myQCustomQWidget.sizeHint())
		
		nums = self.listWidget.count()

		self.listWidget.insertItem(0, myQListWidgetItem)
		self.listWidget.setItemWidget(myQListWidgetItem,myQCustomQWidget)

		if nums > 6:
			print(nums)
			item = self.listWidget.takeItem(nums-1)
			item = None

	def draw(self, image, tracking_objects):
		cv2.line(image, self.crossline[0], self.crossline[1], (0, 255, 255), 2)

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

				cv2.putText(image, tracking_object.mask, (bbox[0]+5, bbox[1]-20),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
				cv2.putText(image, '%s, %d' %(tracking_object.gender, tracking_object.age) , (bbox[0]+5, bbox[1]-5),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

	def displayImage(self, img):
		qformat=QImage.Format_Indexed8
		if len(img.shape)==3:
			if(img.shape[2])==4:
				qformat=QImage.Format_RGBA888
			else:
				qformat=QImage.Format_RGB888
		img = QImage(img,img.shape[1],img.shape[0],qformat)
		img = img.rgbSwapped()
		self.imglabel.setPixmap(QPixmap.fromImage(img))
		self.imglabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

	def exit(self):
		if self.collector is not None:
			self.collector.save()

		self.status = 'stopping'
		self.close()


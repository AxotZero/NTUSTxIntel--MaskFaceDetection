import argparse
import cv2
import logging
import sys

from detector import Detector

from PyQt5.QtWidgets import QApplication
from gui.TrackMachine import Gui


def logging_initialize():
	logging.basicConfig(level=logging.INFO)


def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('-f', '--face_threshold', type=float, default=0.5, 
						help='the face detection threshold')

	parser.add_argument('-m', '--mode', type=str, choices=['crossline', 'firstface'], default='firstface')
	
	parser.add_argument('--input_file', type=str, default='',
					 help='test file path')
	
	parser.add_argument('--save_path', type=str, default='',
                        help='result path')
	
	args = parser.parse_args()
	return args


if __name__ == "__main__":
	logging_initialize()
	args = parse_args()
	app =  QApplication(sys.argv)

	detector = Detector()

	window = Gui(detector, args)
	window.show()

	sys.exit(app.exec_())
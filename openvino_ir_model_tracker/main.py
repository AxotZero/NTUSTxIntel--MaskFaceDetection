import argparse
import cv2
import logging
import sys

from detector import Detector

from PyQt5.QtWidgets import QApplication
from TrackMachine import Gui
from data_collector import DataCollector

def logging_initialize():
	logging.basicConfig(level=logging.DEBUG)


def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('-f', '--face_threshold', type=float, default=0.5, 
						help='the face detection threshold')

	parser.add_argument('--input_file', type=str, default='',
					 help='test_video path')
	
	parser.add_argument('--save_video', type=str, default='',
                        help='save_video path')

	parser.add_argument('--save_data', type=str, default='',
                        help='save_collected_data path')
	
	args = parser.parse_args()
	return args


if __name__ == "__main__":
	logging_initialize()
	args = parse_args()
	app =  QApplication(sys.argv)

	detector = Detector()

	if args.save_data != '':
		collector = DataCollector(args.save_data)
	else:
		collector = None

	window = Gui(detector, collector, args)
	window.show()

	sys.exit(app.exec_())
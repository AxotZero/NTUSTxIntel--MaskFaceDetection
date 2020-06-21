import cv2

vid = cv2.VideoCapture('test_data/drom1.mp4')
vid.set(cv2.CAP_PROP_FRAME_WIDTH,640)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
while True:
	ret, frame = vid.read()
	if not ret:
		print('cannot read frame')
		break

	cv2.imshow('frame', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

vid.release()
cv2.destroyAllWindows()
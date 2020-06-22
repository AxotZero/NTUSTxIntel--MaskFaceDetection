

def cross(p1, p2, p3):
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
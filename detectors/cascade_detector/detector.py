import cv2, sys, os
import numpy as np

class Detector:
	# This example of a detector detects faces. However, you have annotations for ears!

	cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cascades', 'haarcascade_mcs_leftear.xml'))
	# cascade = cv2.CascadeClassifier("cascades/haarcascade_mcs_leftear.xml")
	cascade2 = cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cascades', 'haarcascade_mcs_rightear.xml'))

	def detect(self, img):
		det_list = self.cascade.detectMultiScale(img, 1.05, 1)
		det_list2 = self.cascade2.detectMultiScale(img, 1.05, 1)

		res = []
		for el in det_list:
			res.append(el)

		for el in det_list2:
			res.append(el)

		return np.array(res)


if __name__ == '__main__':
	fname = sys.argv[1]
	img = cv2.imread(fname)
	detector = Detector()
	detected_loc = detector.detect(img)
	for x, y, w, h in detected_loc:
		cv2.rectangle(img, (x,y), (x+w, y+h), (128, 255, 0), 4)
	cv2.imwrite(fname + '.detected.jpg', img)
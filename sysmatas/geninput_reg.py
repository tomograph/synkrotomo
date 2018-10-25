import numpy as np

def printList(ls, buffer):
	buffer += "["
	for item in ls:
		buffer += str(item) 
		buffer += ","
	buffer = buffer.strip(",")
	buffer += "]"

	return buffer;

angle_step = 0.25
angles = np.arange(40.0, 44.0, angle_step)
#angles = np.array([2o9.0])
lbAngles = len(angles) 
#pm1 = Projection_Matrix.line_parallel_tomo(detector, grid, angles)

n = 500


delta = 1.0
# no limited field of view
grid = (delta, n)
#numDetectors = 2
#detectorHalfWidth = 5
detector = (2000, 1000)   
gridHalfWidth = 1000

detectorPositions = np.linspace(-1000, 1000, 2000)
#print detectorPositions
#detectorPositions = np.array([150.0])
numDetectors = len(detectorPositions)
rowOffset = 0

buf = ""
buf = printList(detectorPositions, buf)
buf += " "
buf = printList(angles, buf)
buf += " " + str(gridHalfWidth) 
buf += " " + str(delta)
buf += " " + str(rowOffset)
buf += " " + str(1) + "i16" # 1 for csr
buf += " " + str(len(angles)*len(detectorPositions))
print buf

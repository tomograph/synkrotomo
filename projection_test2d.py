from futhark import backprojection
from futhark import forwardprojection
from skimage.io import imread
from skimage.transform import rotate
import pylab
import sys
import numpy as np

def sinogram(image, theta):
    #sinogram = radon(image, theta, circle=False)
    sinogram = np.zeros((len(theta), max(image.shape)))
    for i in range(0, len(theta)):
        rotated_image = rotate(image, theta[i], resize=False)
        sinogram[i] = sum(rotated_image)
    return sinogram.flatten()

def testbackprojection2D(angles, rays, image):
    s = sinogram(image,angles)
    proj = backprojection.backprojection()
    bp = proj.main(rays.astype(np.float32), angles.astype(np.float32), s.astype(np.float32), image.shape[0])
    pylab.imsave("backprojection.png",np.array(bp.get()).reshape(image.shape))

def testforwardprojection2D(angles, rays, image):
    proj = forwardprojection.forwardprojection()
    bp = proj.main(rays.astype(np.float32), angles.astype(np.float32), image.flatten().astype(np.float32))
    pylab.imsave("forwardprojection.png",np.array(bp.get()).reshape((len(angles),len(rays))))

def main(argv):
    #make phantom with off centered square
    I = np.zeros((256, 256))
    I[133:142, 148:187] = 1 ;
    #initialize values
    size = I.shape[0]
    startvalue = (size-1)/2.0
    rays = np.linspace((-(1.0)*startvalue), startvalue, size)
    angles = np.linspace(0,180, 15, endpoint=False)
    testbackprojection2D(angles, rays, I)
    testforwardprojection2D(angles, rays, I)

if __name__ == '__main__':
    main(sys.argv)

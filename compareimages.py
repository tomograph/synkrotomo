from skimage.measure import compare_ssim as ssim
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.transform import resize

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])

	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def compare_images(imageA, imageB, title):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	s = ssim(imageA, imageB)

    	# setup the figure
	fig = plt.figure(title)
	plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))

	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA, cmap = plt.cm.gray)
	plt.axis("off")

	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB, cmap = plt.cm.gray)
	plt.axis("off")

	# show the images
	plt.show()

#load images
original = imread('C:\\Users\\zgb585\\Documents\\ASTRA\\beatingastra\\output\\original.png')
original = original[::-1,:]
original = resize(original,(256,256))
fbp = imread('C:\\Users\\zgb585\\Documents\\ASTRA\\beatingastra\\output\\astrasfbp_fullsweep.png')
fbp = fbp[38:-38,38:-38]
fbp = resize(fbp,(256,256))
sirt = imread('C:\\Users\\zgb585\\Documents\\ASTRA\\beatingastra\\output\\astrasirt_fullsweep.png')
sirt = sirt[38:-38,38:-38]
sirt = resize(sirt,(256,256))
tenanglesfbp = imread('C:\\Users\\zgb585\\Documents\\ASTRA\\beatingastra\\output\\10angles_full\\astrasfbp.png')
tenanglesfbp = tenanglesfbp[38:-38,38:-38]
tenanglesfbp = resize(tenanglesfbp,(256,256))
tenanglessirt = imread('C:\\Users\\zgb585\\Documents\\ASTRA\\beatingastra\\output\\10angles_full\\astrasirt.png')
tenanglessirt = tenanglessirt[38:-38,38:-38]
tenanglessirt = resize(tenanglessirt,(256,256))
randomanglesfbp = imread('C:\\Users\\zgb585\\Documents\\ASTRA\\beatingastra\\output\\150random_full\\astrasfbp.png')
randomanglesfbp = randomanglesfbp[38:-38,38:-38]
randomanglesfbp = resize(randomanglesfbp,(256,256))
randomanglessirt = imread('C:\\Users\\zgb585\\Documents\\ASTRA\\beatingastra\\output\\150random_full\\astrasirt.png')
randomanglessirt = randomanglessirt[38:-38,38:-38]
randomanglessirt = resize(randomanglessirt,(256,256))
thirtyanglesfbp = imread('C:\\Users\\zgb585\\Documents\\ASTRA\\beatingastra\\output\\30angles_full\\astrasfbp.png')
thirtyanglesfbp = thirtyanglesfbp[38:-38,38:-38]
thirtyanglesfbp = resize(thirtyanglesfbp,(256,256))
thirtyanglessirt = imread('C:\\Users\\zgb585\\Documents\\ASTRA\\beatingastra\\output\\30angles_full\\astrasirt.png')
thirtyanglessirt = thirtyanglessirt[38:-38,38:-38]
thirtyanglessirt = resize(thirtyanglessirt,(256,256))

# initialize the figure
fig = plt.figure("Images")
images = ("Original", original), ("FBP", fbp), ("SIRT", sirt), ("30FBP", thirtyanglesfbp), ("30SIRT", thirtyanglessirt), ("RandFBP", randomanglesfbp), ("RandSIRT", randomanglessirt)

# loop over the images
for (i, (name, image)) in enumerate(images):
	# show the image
	ax = fig.add_subplot(1, 7, i + 1)
	ax.set_title(name)
	plt.imshow(image, cmap=plt.cm.Greys)
	plt.axis("off")

# show the figure
plt.show()

# compare the images
compare_images(original, fbp, "Original vs. FBP")
compare_images(original, sirt, "Original vs. SIRT")
compare_images(original, thirtyanglesfbp, "Original vs. 30 angles FBP")
compare_images(original, thirtyanglessirt, "Original vs. 30 angles SIRT")
compare_images(original, randomanglesfbp, "Original vs. 150 random angles FBP")
compare_images(original, randomanglessirt, "Original vs. 150 random angles SIRT")

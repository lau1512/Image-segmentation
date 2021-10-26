#Input images must be JPEG
%matplotlib inline
import os
import matplotlib as mpl
mpl.rcParams['figure.dpi']=100
import numpy as np
from imageio import imread, imwrite
from skimage.transform import rescale
from skimage.color import rgb2lab, lab2rgb
from skimage import img_as_ubyte
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from PIL import Image

def cluster_assignments(X, Y):
        return np.argmin(euclidean_distances(X,Y), axis=1)
  border_alg = True #Toggle which model to use: The border model or the "most-color" model
input_dir = "test_images" #This input should specify the location of the input directory, containing the images to use
ouput_dir = "test_images_out" #The name of the desired output destination

if not os.path.exists(ouput_dir):
    os.mkdir(ouput_dir)
    
for filename in os.listdir(input_dir):
    print(filename)
    if filename.endswith(".jpg"): #Check if the file appears to be a JPEG file, else skip it
        image_raw = imread(os.path.join(input_dir,filename))
        image_width = 1000
        image = rescale(image_raw, image_width/image_raw.shape[0], mode='reflect', multichannel=True, anti_aliasing=True)
        shape = image.shape
        X = rgb2lab(image).reshape(-1, 3)
        image_c = [image.reshape(-1, 3)[i, :] for i in range(image.shape[0] * image.shape[1])]

        K = 5
        centers = np.array([X.mean(0) + (np.random.randn(3)/10) for _ in range(K)])
        y_kmeans = cluster_assignments(X, centers)

        # repeat estimation a number of times (could do something smarter, like comparing if clusters change)
        for i in range(30):
            # assign each point to the closest center
            y_kmeans = cluster_assignments(X, centers)

            # move the centers to the mean of their assigned points (if any)
            for i, c in enumerate(centers):
                points = X[y_kmeans == i]
                if len(points):
                    centers[i] = points.mean(0)
        X_reduced = lab2rgb(centers[y_kmeans,:].reshape(shape[0], shape[1], 3))

        if border_alg:
            border_arr = np.concatenate([X_reduced[:,0],X_reduced[:,X_reduced.shape[1]-1],X_reduced[0,:],X_reduced[X_reduced.shape[2]-1,:]]) #Create a new array consisting of the pixels from the border of the image
            values, counts = np.unique(border_arr, return_counts=True, axis=0)
        else:
            values, counts = np.unique(X_reduced, return_counts=True, axis=0)
        ind = np.argmax(counts)
        backround_color=values[ind] #Find the most frequent pixel color

        mask_arr = X_reduced != backround_color #Create a mask array thats false for in any position of a pixel that matches the background color and true elsewhere
        result = np.dstack([image,mask_arr[:,:,0]]) #Combine the mask array and the image array to add a 4th color channel (Esentially converting the image to RGBA)
        imwrite(os.path.join(ouput_dir,filename.replace(".jpg",".png")), img_as_ubyte(result), format="PNG-PIL") #Save the image as PNG

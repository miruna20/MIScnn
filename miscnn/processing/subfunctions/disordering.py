"""
A disordering Subfunction class which disorders the context
i.e for n iterations, in every iteration m patches of dimensions x,y,z are swaped pairwise

"""

# External libraries
import numpy as np
import nibabel as nib
# Internal libraries/scripts
from miscnn.processing.subfunctions.abstract_subfunction import Abstract_Subfunction

class Disordering(Abstract_Subfunction):
    def __init__(self,mode="context_restoration"):
        self.mode = mode

    def preprocessing(self,sample,training=True):
        # Access image
        image_twice = sample.img_data
        #practic 96,128,128,0 e o poza
        #96,128,128,1 e cealalta, ma rog ambele poze sunt la fel

        #reshape in a photo with only 3 dimensions
        image = image_twice[:,:,:,0]

        shuffled = self.shuffle(image)

        #only for debugging save the image to see how the disordering looked like
        # save the newly created image
        new_image = nib.Nifti1Image(shuffled,None)
        nib.save(new_image,"/home/temp/shuffled_img/shuffled.nii.gz")

        #staple again the shuffled images
        temp1 = np.reshape(shuffled,[image.shape[0],image.shape[1],image.shape[2],1])
        temp2 = np.reshape(shuffled,[image.shape[0],image.shape[1],image.shape[2],1])
        stapled = np.concatenate((temp1,temp2),axis = 3)

        return stapled

    def postprocessing(self,prediction):
        return prediction

    def pixelCommon(self,leftCorner1, leftCorner2, radius):
        diffx = leftCorner1[0] - leftCorner2[0]
        diffy = leftCorner1[1] - leftCorner2[1]
        diffz = leftCorner1[2] - leftCorner2[2]
        return (diffx * diffx + diffy * diffy + diffz * diffz < radius * radius)

    def swapPatches(self,patch1_leftcorner, patch2_leftcorner, img_nparray, x, y, z):
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    img_nparray[[patch1_leftcorner[0] + i, patch2_leftcorner[0] + i], [patch1_leftcorner[1] + j,
                                                                                       patch2_leftcorner[1] + j], [
                                    patch1_leftcorner[2] + k, patch2_leftcorner[2] + k]] = \
                        img_nparray[[patch2_leftcorner[0] + i, patch1_leftcorner[0] + i], [patch2_leftcorner[1] + j,
                                                                                           patch1_leftcorner[1] + j], [
                                        patch2_leftcorner[2] + k, patch1_leftcorner[2] + k]]

        return img_nparray

    def shuffle(self,image, iter=30, x=30, y=30, z=30):
        # iterations of the shuffling algorithm
        T = iter
        for i in range(T):
            # print("Iteration" + str(i))
            # randomly select a 3D patch p1 (through choosing the left down corner)
            patch1_leftcorner = (np.random.randint(0, image.shape[0] - x), np.random.randint(0, image.shape[1] - y),
                                 np.random.randint(0, image.shape[2] - z))
            print(patch1_leftcorner)

            # randomly select a 3D patch p2 (through choosing the left down corner)
            patch2_leftcorner = (np.random.randint(0, image.shape[0] - x), np.random.randint(0, image.shape[1] - y),
                                 np.random.randint(0, image.shape[2] - z))
            # print(patch2_leftcorner)

            # pick random until the matches do not overlap
            while (self.pixelCommon(patch1_leftcorner, patch2_leftcorner, radius=x + 1)):
                # randomly select a 3D patch  p1
                patch1_leftcorner = (np.random.randint(0, image.shape[0] - x), np.random.randint(0, image.shape[1] - y),
                                     np.random.randint(0, image.shape[2] - z))
                # print(patch1_leftcorner)
                # randomly select a 3D patch  p1
                patch2_leftcorner = (np.random.randint(0, image.shape[0] - x), np.random.randint(0, image.shape[1] - y),
                                     np.random.randint(0, image.shape[2] - z))
                # print(patch2_leftcorner)

            # print(patch1_leftcorner)
            # print(patch2_leftcorner)
            # swap the 2 patches
            image = self.swapPatches(patch1_leftcorner, patch2_leftcorner, image, x, y, z)

        return image


    

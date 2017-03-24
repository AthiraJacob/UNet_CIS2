# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.

'''
author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

#import cv2
import glob
import numpy as np
from PIL import Image
import cv2

class BaseDataProvider(object):
    """
    Abstract base class for DataProvider implementation. Subclasses have to
    overwrite the `_next_data` method that load the next data and label array.
    This implementation automatically clips the data with the given min/max and
    normalizes the values to (0,1]. To change this behavoir the `_process_data`
    method can be overwritten. To enable some post processing such as data
    augmentation the `_post_process` method can be overwritten.

    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping

    """
    
    channels = 3
    n_class = 2
    

    def __init__(self, complexity = 0, a_min=None, a_max=None):
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf
        self.complexity = complexity

    def _load_data_and_label(self):
    	# Loads single image and processes it
        data, label = self._next_data()
            
        train_data = self._process_data(data)
        labels = self._process_labels(label)
        
        train_data, labels = self._post_process(train_data, labels)
        
        nx = data.shape[1]
        ny = data.shape[0]

        return train_data.reshape(1, ny, nx, self.channels), labels.reshape(1, ny, nx, self.n_class),
    
    def _process_labels(self, label):
    	#Create two separate masks with each label
        if self.n_class == 2:
            nx = label.shape[1]
            ny = label.shape[0]
            # label = label/np.max(label)
            
            
            labels = np.zeros((ny, nx, self.n_class), dtype=np.float32)
            labels[..., 1] = label
            labels[..., 0] = ~label
            return labels
        
        return label
    
    def _process_data(self, data):
        # normalization
        data = np.clip(np.fabs(data), self.a_min, self.a_max)
        data -= np.amin(data)
        data /= np.amax(data)
        return data
    
    def _post_process(self, data, labels):
        """
        Post processing hook that can be used for data augmentation
        
        :param data: the data array
        :param labels: the label array
        """

        # No post-augmentation as of now
        return data, labels
    
    def __call__(self, n):
    	# Makes a batch by calling single images in a loop
        train_data, labels = self._load_data_and_label()
        nx = train_data.shape[1]
        ny = train_data.shape[2]
    
        X = np.zeros((n, nx, ny, self.channels))
        Y = np.zeros((n, nx, ny, self.n_class))
    
        X[0] = train_data
        Y[0] = labels
        for i in range(1, n):
            train_data, labels = self._load_data_and_label()
            X[i] = train_data
            Y[i] = labels
    
        return X, Y
    


class ImageDataProvider(BaseDataProvider):
    """
    Generic data provider for images, supports gray scale and colored images.
    Assumes that the data images and label images are stored in the same folder
    and that the labels have a different file suffix 
    e.g. 'train/fish_1.tif' and 'train/fish_1_mask.tif'

    Usage:
    data_provider = ImageDataProvider("..fishes/train/*.tif")
        
    :param search_path: a glob search pattern to find all data and label images
    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping
    :param data_suffix: suffix pattern for the data images. Default '.tif'
    :param mask_suffix: suffix pattern for the label images. Default '_mask.tif'
    
    """
    
    n_class = 2
    
    def __init__(self, search_path, complexity, a_min=None, a_max=None, data_suffix=".png", mask_suffix='_mask.png'):
        super(ImageDataProvider, self).__init__(complexity,a_min, a_max)
        self.data_suffix = data_suffix
        self.mask_suffix = mask_suffix
        self.fg_idx = -1
        self.bg_idx = -1
        self.fg_files = self._find_data_files(search_path + '/fg_trial')
        self.bg_files = []
        if complexity == 0:
            self.bg_folders = self._find_data_files(search_path + '/bg_trial')
            for i in range(len(self.bg_folders)):
                self.bg_files = self.bg_files+ self._find_data_files(search_path + '/bg_trial/bg_' + str(i+1) + '/good lighting')
                self.bg_files = self.bg_files+ self._find_data_files(search_path + '/bg_trial/bg_' + str(i+1) + '/bad lighting')
        else:
        	self.bg_files = self._find_data_files(search_path + '/bg_trial/bg_' + str(complexity))
    
        assert len(self.fg_files) > 0, "No foreground files"
        assert len(self.bg_files) > 0, "No background files"
        print("Number of foreground files used: %s" % len(self.fg_files))
        print("Number of background files used: %s" % len(self.bg_files))
        img = self._load_file(self.fg_files[0])
        self.channels = 1 if len(img.shape) == 2 else img.shape[-1]
        
    def _find_data_files(self, search_path):
        all_files = glob.glob(search_path+'/*')
        return [name for name in all_files if not self.mask_suffix in name]
    
    
    def _load_file(self, path, dtype=np.float32):
        # return np.array(Image.open(path), dtype)
        return np.squeeze(cv2.imread(path))

    def _cylce_file(self):
        self.fg_idx += 1
        self.bg_idx += 1
        if self.fg_idx >= len(self.fg_files):
            self.fg_idx = 0
        if self.bg_idx >= len(self.bg_files):
            self.bg_idx = 0 
     
    def _segment(self,img):
    	b, g, r = cv2.split(img)
    	ret, mask = cv2.threshold(r, 0, 255, cv2.THRESH_OTSU)
    	mask = 255 - mask

    	kernel = np.ones((10, 10), np.uint8)
    	closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    	closing = closing == 255
    	#Convert mask to be of the same shape as input image
    	# closing = closing.reshape(list(closing.shape)+[1])
    	# closing = np.repeat(closing, 3, axis=2)
    	return closing

    def _join(self,fg, bg, mask):
	'''
	Join masked fg image and bg image by zeroing out masked parts fg, 
	the complement in bg, and adding them both
	'''
	#If mask is single channel, convert to 3 channel
        if mask.ndim!= fg.ndim or  mask.ndim!= bg.ndim:  
		  mask = mask.reshape(list(mask.shape)+[1])
		  mask = np.repeat(mask, 3, axis=2)
        masked_fg = np.multiply(fg, mask)
        masked_bg = np.multiply(bg, 1.0 - mask)
        joined = np.add(masked_fg, masked_bg)   
        return joined

    def _augment(self,img,prob,mask = None):
		'''
		Function to transform fg, bg: Rotate by random angle, horizontal/vertical flips, translation (only bg), zoom
		color jitter? 
		Inputs: img: input image (foreground/background)
				prob: augment image with probability prob
				mask: provided if img is foreground image
		Output: nimg: transformed image
				nmask: transformed mask (only in case of foreground images)
		'''
		nimg = img
		if mask is not None:
			nmask = mask*1.0

		if np.random.rand()<prob:   # Vertical flip
			nimg = cv2.flip(nimg,0)
			if mask is not None:
				nmask = cv2.flip(nmask,0)

		if np.random.rand()<prob:   # Horizontal flip
			nimg = cv2.flip(nimg,1)
			if mask is not None:
				nmask = cv2.flip(nmask,1)

		if np.random.rand()<prob:   # Rotate by random angle
			rows,cols = nimg.shape[0],nimg.shape[1]
			angle = np.random.rand()*180;
			M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
			nimg = cv2.warpAffine(nimg,M,(cols,rows))
			if mask is not None:
				nmask = cv2.warpAffine(nmask,M,(cols,rows))

		if np.random.rand() < prob: #translation by random distance
			rows,cols = nimg.shape[0],nimg.shape[1]
			shift = (np.random.rand() - 0.5)*0.5*rows, (np.random.rand() - 0.5)*0.5*cols
			M = np.float32([[1,0,shift[0]],[0,1,shift[1]]])
			nimg = cv2.warpAffine(nimg,M,(cols,rows))
			nmask = cv2.warpAffine(nmask,M,(cols,rows))

		if np.random.rand() < prob: #Zoom in and crop. Zooms in to maximum 2 times
			rows,cols = nimg.shape[0],nimg.shape[1]
			zoom_factor = np.random.rand() + 1
			nimg = cv2.resize(nimg,None,fx=zoom_factor, fy=zoom_factor, interpolation = cv2.INTER_LINEAR)
			x1,y1 = int((nimg.shape[0] - rows)/2), int((nimg.shape[1] - cols)/2)
			nimg = nimg[x1:x1 + rows, y1:y1+cols]
			if mask is not None:
				nmask = cv2.resize(nmask,None,fx=zoom_factor, fy=zoom_factor, interpolation = cv2.INTER_LINEAR)
				nmask = nmask[x1:x1 + rows, y1:y1+cols]
		
		if mask is not None:
			return nimg,nmask
		else:
			return nimg


    def _compose(self,fg_img,bg_img, mask = None, prob_fg = 0.5, prob_bg = 0):
		'''
		Function to compose foreground and background into an image (includes augmentation of foreground).
		Inputs: fg_img: foreground image
			bg_img: background image 
			mask: mask for segmenting fg_img. If None, segment it online (not recommended!) 
		Ouput:  composed_img:    composed image
				mask_augmented = corresponding mask
		'''
		
		# Get the mask of the foreground and make it to 3 channels
		if mask is None:
			mask = self._segment(fg_img)
		composed_imgs = np.zeros(bg_img.shape)
		fg_augmented,mask_augmented = self._augment(fg_img,prob_fg,mask = mask)
		bg_augmented = self._augment(bg_img,prob_bg,mask = None)
		composed_img = self._join(fg_augmented,bg_augmented, mask_augmented)	
		return composed_img,mask_augmented   

    def _next_data(self):
        self._cylce_file()
        fg_image_name = self.fg_files[self.fg_idx]
        bg_image_name = self.bg_files[self.bg_idx]
        label_name = fg_image_name.replace(self.data_suffix, self.mask_suffix)
        
        fg_img = self._load_file(fg_image_name, np.float32)
        bg_img = self._load_file(bg_image_name, np.float32)
        label_img = self._load_file(label_name)
        img, label = self._compose(fg_img,bg_img, mask = label_img, prob_fg = 0.5, prob_bg = 0)
        # label = self._load_file(label_name, np.bool)
    	if label.ndim != 2:
            label = label[:,:,0]
            # label = label/np.max(label)
            # label[label<0.1]=0
            # label[label>0.9] = 1
        ret,label = cv2.threshold(cv2.convertScaleAbs(label),0.9,1,cv2.THRESH_BINARY)
        return img,label

import cv2
import os, random, string
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
from PIL import ImageFont, ImageDraw, Image
from parameter import *

sometimes = lambda aug: iaa.Sometimes(0.5, aug)
seq = iaa.Sequential(
		[
			# apply the following augmenters to most images
			# crop images by -5% to 10% of their height/width
			sometimes(iaa.CropAndPad(
				percent=(-0.01, 0.1),
				pad_mode=['constant', 'edge', 'median'],
				pad_cval=(0, 255)
			)),
			sometimes(iaa.Affine(
				scale={"x": (0.95, 1.05), "y": (0.95, 1.05)}, # scale images to 95-105% of their size, individually per axis
				translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, # translate by -5 to +5 percent (per axis)
				rotate=(-5, 5), # rotate by -5 to +5 degrees
				shear=(-8, 8), # shear by -8 to +8 degrees
				order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
				cval=(0, 255), # if mode is constant, use a cval between 0 and 255
				mode=['constant', 'edge']# use any of scikit-image's warping modes 
			)),

			iaa.SomeOf((0, 5),
				[
					iaa.OneOf([
						iaa.GaussianBlur((0, 2.0)), # blur images with a sigma between 0 and 2.0
						iaa.AverageBlur(k=(2, 4)), # blur image using local means with kernel sizes between 2 and 4
						iaa.MedianBlur(k=(3,5)), # blur image using local medians with kernel sizes between 3 and 5
					]),
					iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
					iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images

					# search either for all edges or for directed edges,
					# blend the result with the original image using a blobby mask
					iaa.SimplexNoiseAlpha(iaa.OneOf([
						iaa.EdgeDetect(alpha=(0.3, 0.7)),
						iaa.DirectedEdgeDetect(alpha=(0.3, 0.7), direction=(0.0, 0.7)),
					])),

					iaa.OneOf([
						iaa.Dropout((0.01, 0.05), per_channel=0.1), # randomly remove up to 10% of the pixels
						# iaa.CoarseDropout((0.03, 0.05), size_percent=(0.01, 0.03), per_channel=0.1),
					]),

					# either change the brightness of the whole image (sometimes
					# per channel) or change the brightness of subareas
					iaa.OneOf([
						iaa.FrequencyNoiseAlpha(
							exponent=(-2, 0),
							first=iaa.Multiply((0.5, 1.5), per_channel=True),
							second=iaa.LinearContrast((0.5, 2.0))
						)
					]),
					iaa.LinearContrast((0.5, 1.0), per_channel=0.03), # improve or worsen the contrast
					sometimes(iaa.ElasticTransformation(alpha=(0.5, 2.0), sigma=0.15)), # move pixels locally around (with random strengths)
					sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.02))), # sometimes move parts of the image around
					sometimes(iaa.PerspectiveTransform(scale=0.05, keep_size=True))
				],
				random_order=True
			)
		],
		random_order=True
	)



class TextImageGenerator(object):

	def __init__(self,batch_size,input_shape = (64,300),fonts_location = 'freefont/'):
		self.fonts_location = fonts_location
		self.batch_size = batch_size
		self.CHARS = CHARS
        #self.input_shape = input_shape

		self.CHARS_DICT = {char:i for i, char in enumerate(self.CHARS)}
		self.NUM_CHARS = len(self.CHARS)
		
	def get_separator(self,fields = [""," "]):
		return random.choice(fields)

	def encode(self,s):
		label = np.zeros(shape=(max_text_len))
		for i,c in enumerate(s):
			label[i]=self.CHARS.index(c)
		return label

	def image_generator(self,state_code='KA',separator = [""," "] ):
		"""
		its a generator
		each call will give an image of random number plate with
		random image augmentation .
		"""
		fonts = [ImageFont.truetype(os.path.join(self.fonts_location,font_type),random.randint(32,40)) for font_type in os.listdir(self.fonts_location) ]

		font = random.choice(fonts) 
		single_line = True #random.choice([True,False])
		if single_line:
			number_plate = state_code+self.get_separator(separator)+str(random.randint(0,9))+str(random.randint(0,9))+\
			self.get_separator(separator)+random.choice(" "+string.ascii_uppercase)+random.choice(string.ascii_uppercase)+\
			self.get_separator(separator)+str(random.randint(0,9))+str(random.randint(0,9))+str(random.randint(0,9))+str(random.randint(0,9))
			if len(number_plate)<13:
				t = len(number_plate)
				for i in range(13-t):
					number_plate = number_plate+" "
		else :
			separator = separator +["\n"]
			number_plate = state_code+self.get_separator(separator)+str(random.randint(0,9))+str(random.randint(0,9))+\
			self.get_separator(separator)+random.choice(" "+string.ascii_uppercase)+random.choice(string.ascii_uppercase)+\
			self.get_separator(separator)+str(random.randint(0,9))+str(random.randint(0,9))+str(random.randint(0,9))+str(random.randint(0,9))
			#Creating an image with height,width  and color
		img = Image.new('RGB', (300,64), (random.randint(0,255),random.randint(0,255),random.randint(0,255)))
		
		draw = ImageDraw.Draw(img)
		draw.text((15,12), number_plate, (0,0,0), font=font)
		data =np.asarray(img)

		data = np.transpose(data,(1,0,2))

		return data,number_plate

	def next_batch(self):
		while True:
			op = [self.image_generator() for i in range(self.batch_size)]
			input_length = np.ones((self.batch_size, 1)) * 30  
			label_length = np.zeros((self.batch_size, 1))
			input_X = seq(images = np.array([i[0] for i in op]))
			input_X = (input_X/255.0)  
			
			input_Y = np.array([self.encode(i[1]) for i in op])
			
			for i in range(len(op)):
				label_length[i]=len(op[i][1])


			inputs = {
					'the_input': input_X,  
					'the_labels': input_Y,  
					'input_length': input_length,  
					'label_length': label_length  
					}
			outputs = {'ctc': np.zeros([self.batch_size])}   
				
			yield (inputs, outputs)
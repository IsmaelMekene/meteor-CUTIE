"""
The implementation of Data Generator based on Tensorflow.

@Author: Mékéné
@Github: https://github.com/IsmaelMekene
@Project: https://github.com/luyanger1799/meteor-CUTIE

"""
from tensorflow.python.keras.preprocessing.image import Iterator
from keras_applications import imagenet_utils
from utils.utils import *
import tensorflow as tf
import numpy as np

keras_utils = tf.keras.utils


class DataIterator(Iterator):
    def __init__(self,
                 image_data_generator,
                 images_list,
                 labels_list,
                 num_classes,
                 batch_size,
                 target_size,
                 shuffle=True,
                 seed=None,
                 data_aug_rate=0.):
        num_images = len(images_list)

        self.image_data_generator = image_data_generator
        self.images_list = images_list
        self.labels_list = labels_list
        self.num_classes = num_classes
        self.target_size = target_size
        self.data_aug_rate = data_aug_rate

        super(DataIterator, self).__init__(num_images, batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(shape=(len(index_array),) + self.target_size + (3,))
        batch_y = np.zeros(shape=(len(index_array),) + self.target_size + (self.num_classes,))

        for i, idx in enumerate(index_array):
            image, label = load_image(self.images_list[idx]), load_image(self.labels_list[idx])
            # random crop
            if self.image_data_generator.random_crop:
                image, label = random_crop(image, label, self.target_size)
            else:
                image, label = resize_image(image, label, self.target_size)
            # data augmentation
            if np.random.uniform(0., 1.) < self.data_aug_rate:
                # random vertical flip
                if np.random.randint(2):
                    image, label = random_vertical_flip(image, label, self.image_data_generator.vertical_flip)
                # random horizontal flip
                if np.random.randint(2):
                    image, label = random_horizontal_flip(image, label, self.image_data_generator.horizontal_flip)
                # random brightness
                if np.random.randint(2):
                    image, label = random_brightness(image, label, self.image_data_generator.brightness_range)
                # random rotation
                if np.random.randint(2):
                    image, label = random_rotation(image, label, self.image_data_generator.rotation_range)
                # random channel shift
                if np.random.randint(2):
                    image, label = random_channel_shift(image, label, self.image_data_generator.channel_shift_range)
                # random zoom
                if np.random.randint(2):
                    image, label = random_zoom(image, label, self.image_data_generator.zoom_range)

            image = imagenet_utils.preprocess_input(image.astype('float32'), data_format='channels_last',
                                                    mode='torch')
            label = one_hot(label, self.num_classes)

            batch_x[i], batch_y[i] = image, label

        return batch_x, batch_y


class ImageDataGenerator(object):
    def __init__(self,
                 random_crop=False,
                 rotation_range=0,
                 brightness_range=None,
                 zoom_range=0.0,
                 channel_shift_range=0.0,
                 horizontal_flip=False,
                 vertical_flip=False):
        self.random_crop = random_crop
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.zoom_range = zoom_range
        self.channel_shift_range = channel_shift_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

    def flow(self,
             images_list,
             labels_list,
             num_classes,
             batch_size,
             target_size,
             shuffle=True,
             seed=None,
             data_aug_rate=0.):
        return DataIterator(image_data_generator=self,
                            images_list=images_list,
                            labels_list=labels_list,
                            num_classes=num_classes,
                            batch_size=batch_size,
                            target_size=target_size,
                            shuffle=shuffle,
                            seed=seed,
                            data_aug_rate=data_aug_rate)
    
    
    
    
class DataGeneratorDensAspp(keras.utils.Sequence):
    'Generates data for Keras'


        #'Initialization'
    
    def __init__(self, batch_size, dataframe, input_size = 256, shuffle=True):


      self.batch_size = batch_size
      self.dataframe = dataframe
      self.shuffle = shuffle  #NOTE that the SHUFFLE is at the beginning of each epoch!!!
      self.input_size = input_size
      self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'

        return int(np.floor(len(dataframe) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'

        # Find the kth corresponding batchsize dataframe
        df_temp = dataframe.iloc[index*self.batch_size:(index+1)*self.batch_size, :]

        # Generate data
        larray = self.name_generation(df_temp, self.batch_size)
        
        return larray 


    def on_epoch_end(self):
        'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.dataframe))
        if self.shuffle == True:    #if shuffle is activated
            np.random.shuffle(self.indexes)  #randomly generate the index


    def name_generation(self, batch_size):
      'Generates data names following batch_size samples'

      #k = 19
      n_batch = batch_size  #set number of batch
   


      for k in range(int(len(self.dataframe)/batch_size)):#iterate over the batches 
      
        #dataframe for each batch
        boom = self.dataframe.iloc[k*n_batch:(k+1)*n_batch, :] 

        #empty array of size (batch_size,input,input,3)
        X = np.empty((batch_size, self.input_size, self.input_size, 3))

        for j, pure in enumerate (boom['lesimages']):
          al = plt.imread(pure)

        #resizing the images

          #pure_images.append(al)
          X[j] = al



        #empty array of size (batch_size,input,input,1)
        Y_1 = np.empty((batch_size, int(self.input_size), int(self.input_size), 2))
        
        for zk, noms in enumerate (boom['lesmasks']):

          
          ole = plt.imread(noms)
          

          Y_1[zk] = ole[:,:,0:2]


        '''

        '''

        list_de_sortie = []  #empty list
        list_de_sortie.append(X)  #add the list of images
        list_de_sortie.append(Y_1)   #add the list of the y
        

        yield list_de_sortie






class DataGeneratorPSPNet(keras.utils.Sequence):
    'Generates data for Keras'


        #'Initialization'
    
    def __init__(self, batch_size, dataframe, input_size = 256, shuffle=True):


      self.batch_size = batch_size
      self.dataframe = dataframe
      self.shuffle = shuffle  #NOTE that the SHUFFLE is at the beginning of each epoch!!!
      self.input_size = input_size
      self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'

        return int(np.floor(len(dataframe) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'

        # Find the kth corresponding batchsize dataframe
        df_temp = dataframe.iloc[index*self.batch_size:(index+1)*self.batch_size, :]

        # Generate data
        larray = self.name_generation(df_temp, self.batch_size)
        
        return larray 


    def on_epoch_end(self):
        'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.dataframe))
        if self.shuffle == True:    #if shuffle is activated
            np.random.shuffle(self.indexes)  #randomly generate the index


    def name_generation(self, batch_size):
      'Generates data names following batch_size samples'

      #k = 19
      n_batch = batch_size  #set number of batch
   


      for k in range(int(len(self.dataframe)/batch_size)):#iterate over the batches 
      
        #dataframe for each batch
        bom = self.dataframe.iloc[k*n_batch:(k+1)*n_batch, :] 

        #print(boom['newmasks'])

        #empty array of size (batch_size,input,input,3)
        X = np.empty((batch_size, self.input_size, self.input_size, 3))
        
        #myimages = bom['lesimages'].tolist()

        myimages = bom.iloc[:,0].tolist()

        for j, pure in enumerate(myimages):
          al = plt.imread(pure)
          ali = cv2.resize(al, (480, 480), interpolation=cv2.INTER_NEAREST)

        #resizing the images

          #pure_images.append(al)
          X[j] = ali



        #empty array of size (batch_size,input,input,1)
        Y_1 = np.empty((batch_size, int(self.input_size), int(self.input_size), 1))

        #mymasks = bom['newmasks'].tolist()
        
        mymasks = bom.iloc[:,1].tolist()

        for zk, noms in enumerate(mymasks):

          vide = np.zeros((480,480,1))
          ole = plt.imread(noms)
          aloh = cv2.resize(ole, (480, 480), interpolation=cv2.INTER_NEAREST)
          #print(aloh.shape)
          #print(type(aloh))
          aloh_reshaped = aloh[:,:,0]
          vide[:,:,0] = aloh_reshaped
       
          
          #print(noms)

          Y_1[zk] = vide


        '''

        '''

        list_de_sortie = []  #empty list
        list_de_sortie.append(X)  #add the list of images
        list_de_sortie.append(Y_1)   #add the list of the y
        

        yield list_de_sortie







 
 

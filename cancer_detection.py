import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.application.vgg16 import preprocess_input
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Dropout
from keras.layers import Concatenate
from sklearn.model_selection import train_test_split
from PIL import Image
import sys
from glob import glob

#code to remove forst 150000 images in train directory
#for roots, dirs, filenames in os.walk('/Users/vidushi/Downloads/histopathologic-cancer-detection/train'):
#    for fn in filenames[:150000]:
#        try:
#            os.remove(os.path.join(roots, fn))
#            print("file removed")
#        except:
#            print("unable to remove file")


    
#Preprocess
def append_ext(fn):
    return fn+".tif"

traindf = pd.read_csv('train_labels.csv',dtype = str)

traindf["id"]=traindf["id"].apply(append_ext)

print("labels read completely")
#testdf["id"]=testdf["id"].apply(append_ext)

#gb
from skimage.io import imread
labels = pd.read_csv('train_labels.csv')

# Create `train_sep` directory
train_dir = '/Users/vidushi/Downloads/histopathologic-cancer-detection/train/'
train_sep_dir = 'train_sep/'
if not os.path.exists(train_sep_dir):
    os.mkdir(train_sep_dir)

print("made training directory")

print("Starting copying images")

for filename, class_name in labels.values:
    # Create subdirectory with `class_name`
    if not os.path.exists(train_sep_dir + str(class_name)):
        os.mkdir(train_sep_dir + str(class_name))
    src_path = train_dir + filename + '.tif'
    dst_path = train_sep_dir + str(class_name) + '/' + filename + '.tif'
    try:
        shutil.copy(src_path, dst_path)
    except IOError as e:
        print('Unable to copy file {} to {}'
              .format(src_path, dst_path))
    except:
        print('When try copy file {} to {}, unexpected error: {}'
              .format(src_path, dst_path, sys.exc_info()))


print("Images copying complete")
# # Creating Train / Val / Test folders (One time use)
root_dir = 'train_sep'
posCls = '/1'
negCls = '/0'

os.makedirs(root_dir +'/train' + posCls)
os.makedirs(root_dir +'/train' + negCls)
os.makedirs(root_dir +'/val' + posCls)
os.makedirs(root_dir +'/val' + negCls)
os.makedirs(root_dir +'/test' + posCls)
os.makedirs(root_dir +'/test' + negCls)

# Creating partitions of the data after shuffeling
currentCls = posCls
src = "train_sep"+currentCls # Folder to copy images from

allFileNames = os.listdir(src)
np.random.shuffle(allFileNames)
train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                          [int(len(allFileNames)*0.7), int(len(allFileNames)*0.85)])


train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]
test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]

print('Total images: ', len(allFileNames))
print('Training: ', len(train_FileNames))
print('Validation: ', len(val_FileNames))
print('Testing: ', len(test_FileNames))

# Copy-pasting images
for name in train_FileNames:
    shutil.copy(name, "train_sep/train"+currentCls)

for name in val_FileNames:
    shutil.copy(name, "train_sep/val"+currentCls)

for name in test_FileNames:
    shutil.copy(name, "train_sep/test"+currentCls)
    
#repeat this process with currentCls = negCls to copy past images in 0 sub folder.

print("Copying complete")
    
#Data Augmentation
    
print("Creating training generator")
training_data_generator = ImageDataGenerator(rescale=1./255,
                                             horizontal_flip=True,
                                             vertical_flip=True,
                                             rotation_range=180,
                                             zoom_range=0.5,
                                             width_shift_range=0.4,
                                             height_shift_range=0.3,
                                            channel_shift_range=0.3)

train_generator=training_data_generator.flow_from_directory(
                                            'train_sep/train',
                                            
                                            batch_size=32,
                                            seed=42,
                                            shuffle=True,
                                            class_mode="binary",
                                            target_size=(64,64))


print("Creating test data generator")
test_data_generator=ImageDataGenerator(rescale=1./255)

test_generator=test_data_generator.flow_from_directory(
                                                'train_sep/test',
                                                batch_size=32,
                                                seed=42,
                                                shuffle=False,
                                                class_mode='binary',
                                                target_size=(64,64))


print("Creating validation generator")
validation_data_generator = ImageDataGenerator(rescale=1./255)

validation_generator = validation_data_generator.flow_from_directory(
                                                'train_sep/val',
                                                batch_size=32,
                                                seed=42,
                                                shuffle=False,
                                                class_mode='binary',
                                                target_size=(64,64))

#for removing imagefiles creating an issue(outliers)
print("removing outlier images")
for parent, dirnames, filenames in os.walk('/train_sep/train'):
    for img in filenames:
        if not(img == Image.open(fn)):
            os.remove(os.path.join(parent, fn))
#########CNN MODEL####################################
#print("Creating Neural Network")
##Model
#classifier = Sequential()
##layer1
#classifier.add(Conv2D(32, (3, 3), input_shape = (224, 224 , 3), activation = 'relu'))
#classifier.add(MaxPooling2D(pool_size = (2, 2)))
#classifier.add(Dropout(p = 0.1))
##layer2
#classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
#classifier.add(MaxPooling2D(pool_size = (2, 2)))
#classifier.add(Dropout(p = 0.1))
#
##final
#classifier.add(Flatten())
#classifier.add(Dense(units = 128, activation = 'relu'))
#classifier.add(Dense(units = 1, activation = 'softmax'))
#
#classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#
#
#print("Start training")
#classifier.fit_generator(train_generator,
#                         steps_per_epoch = 8000,
#                         epochs = 10,
#                         validation_data = test_generator,
#                         validation_steps = 2000)

########## TRANSFER LEARNING ##################################
print("starting transfer learning")
#transfer learning
IMAGE_SIZE= [224,224]

vgg = VGG(input_shape = IMAGE_SIZE + [3], weights = 'imagenet', include_top = False)

#existing weights should not be trained
for layer in vgg.layer:
    layer.trainable = False
    
x = Flatten()(vgg.output)

prediction = Dense(2,ctivation = 'softmax')(x)

model = Model(inputs = vgg.input, outputs = prediction)

model.summary()

model.fit_generator(train_generator,
                         steps_per_epoch = 8000,
                         epochs = 10,
                         validation_data = test_generator,
                         validation_steps = 2000)

#Creating a checkpointer
checkpointer = ModelCheckpoint(filepath='train_sep/train', verbose=1 ,save_best_only=True)


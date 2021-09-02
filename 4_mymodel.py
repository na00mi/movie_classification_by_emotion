from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten,GlobalAveragePooling2D, \
    Conv2D, Activation, BatchNormalization, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import timeit
from tqdm.notebook import tqdm



##GPU : 필요한 만큼만 메모리 할당##
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        #print(e)
        pass


train_data_dir='train'
validation_data_dir='validation'
test_data_dir='test'

NUM_CLASSES = 5
IMG_WIDTH, IMG_HEIGHT = 48, 48
batch_size = 32
epoch=100


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rotation_range=20,
                                   # width_shift_range=0.2,
                                   # height_shift_range=0.2,
                                   zoom_range=0.2,
                                   validation_split=0.2
                                   )
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)


train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(IMG_WIDTH,
                                                                 IMG_HEIGHT),
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    seed=12345,
                                                    class_mode='categorical',
                                                    subset='training')

# print('---converting train img to grayscale---')
# train_generator=tf.image.rgb_to_grayscale(train_generator)
# print('---finish---')

validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=batch_size,
    shuffle=False,
    class_mode='categorical',
    subset='validation')

# print('---converting valid img to grayscale---')
# validation_generator=tf.image.rgb_to_grayscale(validation_generator)
# print('---finish---')


# print('!!!!convert finish start training!!!')

#model##
def final_model():
    # inputs=Input(shape=(IMG_WIDTH, IMG_HEIGHT,3))

    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(48,48,3)))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))


    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))
    return model

final_model=final_model()

final_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['acc'],
)


##training##
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=7)
mc = ModelCheckpoint('4_1_mymodel.h5', monitor='val_loss', mode='min', save_best_only=True)

start_time=timeit.default_timer()

history = final_model.fit(
    train_generator,
    steps_per_epoch=7938//batch_size,
    epochs=epoch,
    validation_data=validation_generator,
    validation_steps=1982//batch_size,
    callbacks=[es, mc]
)
# final_model.save('2_9_affectnet_adddata.h5')
# final_model.save_weights('2_3_affectnet_weights.h5')

terminate_time=timeit.default_timer()
print('시작시간: ' , start_time)
print('총 소요시간: ', (terminate_time-start_time)/3600)


##test##
test_generator=val_datagen.flow_from_directory(
    test_data_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=batch_size,
    shuffle=False,
    class_mode='categorical'
)

# test_generator=tf.image.rgb_to_grayscale(test_generator)

final_model.evaluate(test_generator, steps=500//batch_size, verbose=1)
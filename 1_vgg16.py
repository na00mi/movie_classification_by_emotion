from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
import timeit



##GPU : 필요한 만큼만 메모리 할당##
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


train_data_dir='train'
validation_data_dir='validation'
test_data_dir='test'

NUM_CLASSES = 5
IMG_WIDTH, IMG_HEIGHT = 48, 48
batch_size = 32
epoch=100


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
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
validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=batch_size,
    shuffle=False,
    class_mode='categorical',
    subset='validation')



##model##
def final_model():
    #weight부터 새로 학습
    base_model=VGG16(include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT,3))
    base_model.trainable=True

    model=Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model

final_model=final_model()

final_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['acc'],
)


##training##

start_time=timeit.default_timer()

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
mc = ModelCheckpoint('2_vgg16_adddata.h5', monitor='val_loss', mode='min', save_best_only=True)

history = final_model.fit(
    train_generator,
    steps_per_epoch=7938//batch_size,
    epochs=epoch,
    validation_data=validation_generator,
    validation_steps=1982//batch_size,
    callbacks=[es, mc]
)

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

final_model.evaluate(test_generator, steps=500//batch_size, verbose=1)


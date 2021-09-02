import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import glob
import tensorflow as tf


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # 텐서플로가 세 번째 GPU만 사용하도록 제한
    try:
        tf.config.experimental.set_visible_devices(gpus[2], 'GPU')
    except RuntimeError as e:
        # 프로그램 시작시에 접근 가능한 장치가 설정되어야만 합니다
        print(e)

model_name='2_8_final_model.h5'
width, height= 48, 48

model=load_model(model_name)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

pred_img_path='movie_crop_img'
img_name=glob.glob('movie_crop_img/*')

#한국인 감정분류
label={0:'angry',  1:'happy',  2:'neutral',  3:'panic',  4:'sad'}


pred_res=[]

for img in img_name:
    img=img.split('/')[1]
    test=image.load_img(f'{pred_img_path}/{img}', target_size=(width, height,3))
    x = image.img_to_array(test)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict_classes(images, batch_size=10)
    y_pred=model.predict(images)

    print(img) #어떤 이미지인지
    print(classes, label[classes[0]]) # 제일 높은 확률의 label
    print(y_pred) # 전체 클래스에 대한 label
    print('\n')

    pred_res.append({'파일명': img, '감정':label[classes[0]],\
                     '가장 높은 확률': classes, 'angry': y_pred[0][0], 'happy':y_pred[0][1],'neutral':y_pred[0][2], 'panic':y_pred[0][3], 'sad':y_pred[0][4]})

pred_res_df=pd.DataFrame(pred_res)
pred_res_df.to_csv('predict_results/'+model_name+'_pred_res.csv', encoding='utf-8')


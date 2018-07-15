import numpy as np
import pandas as pd
import keras
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten

from tqdm import tqdm
from sklearn.model_selection import train_test_split
import cv2

df_train = pd.read_csv('train_labels.csv')
df_test = pd.read_csv('sample_submission.csv')

targets_series = pd.Series(df_train['invasive'])

one_hot_labels = np.array(targets_series)

im_size = 224

x_train = []
y_train = []
x_test = []

i = 0
for f, invasive in tqdm(df_train.values):
    img = np.array(cv2.imread('train/{}.jpg'.format(f)), dtype=np.float32)
    label = one_hot_labels[i]
    x_train.append(preprocess_input(cv2.resize(img, (im_size, im_size))))
    y_train.append(label)
    i += 1

for f in tqdm(df_test['name'].values):
    img = np.array(cv2.imread('test/{}.jpg'.format(f)), dtype=np.float32)
    x_test.append(preprocess_input(cv2.resize(img, (im_size, im_size))))

y_train_raw = np.array(y_train, np.uint8)
x_train_raw = np.array(x_train, np.float32)
x_test  = np.array(x_test, np.float32)

print(x_train_raw.shape)
print(y_train_raw.shape)
print(x_test.shape)

X_train, X_valid, Y_train, Y_valid = train_test_split(x_train_raw, y_train_raw, test_size=0.3, random_state=1)

base_model = ResNet50(weights=None, include_top=False, input_shape=(im_size, im_size, 3), pooling="max")

x = base_model.output
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = True

model.compile(loss='binary_crossentropy', optimizer='adam',  metrics=['accuracy'])

callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]
model.summary()

model.fit(X_train, Y_train, epochs=10, validation_data=(X_valid, Y_valid), verbose=1)

preds = model.predict(x_test, verbose=1)

sub = pd.DataFrame(preds)

col_names = ['invasive']
sub.columns = col_names

sub.insert(0, 'name', df_test['name'])
sub.head(5)

sub.to_csv("my_submission_base.csv", index=False)





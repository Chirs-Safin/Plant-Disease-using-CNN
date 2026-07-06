import tensorflow as tf
import matplotlib.pyplot as plt


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    BatchNormalization,
    Dropout,
    Rescaling,
    RandomFlip,
    RandomRotation,
    RandomZoom
)
train = tf.keras.utils.image_dataset_from_directory(r"C:\VS code\dataset\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train",image_size=(128,128),batch_size=32,shuffle=True)
valid = tf.keras.utils.image_dataset_from_directory(r"C:\VS code\dataset\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\valid",image_size=(128,128),batch_size=32,shuffle=True)

cln = train.class_names

norm = Rescaling(1./255)
train = train.map(lambda x,y:(norm(x),y))
valid = valid.map(lambda x,y:(norm(x),y))

AUTOTUNE = tf.data.AUTOTUNE
train = train.prefetch(AUTOTUNE)
valid = valid.prefetch(AUTOTUNE)

aug =Sequential([RandomFlip("horizontal"),RandomRotation(0.2),RandomZoom(0.2)])

mod = Sequential([

    aug,

    Conv2D(32,(3,3),padding="same",activation="relu",input_shape=(224,224,3)),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.25),

    Conv2D(64,(3,3),padding="same",activation="relu"),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.25),

    Conv2D(128,(3,3),padding="same",activation="relu"),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.25),

    GlobalAveragePooling2D(),

    Dense(256,activation="relu"),
    Dropout(0.5),

    Dense(38,activation="softmax")

])

mod.compile(optimizer = "adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
mod.summary()

early = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

check = tf.keras.callbacks.ModelCheckpoint(
    "best_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

red = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    verbose=1
)

his=mod.fit(train,validation_data=valid,epochs=15,callbacks=[check,early,red])
mod.save("plant_dis1.keras")
print("training completed")



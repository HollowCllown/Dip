from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

train_dir = 'train'

val_dir = 'val'

test_dir = 'test'

img_width, img_height = 750, 500
input_shape = (img_width, img_height, 3)

#Количество эпох и парарельных изображений
epochs = 20
batch_size = 2

#Распределение датасета
nb_train_samples = 90
nb_validation_samples = 15
nb_test_samples = 15

model = Sequential()
#Первый свёрточный слой с функцией активации и слоем пулинга
model.add(Conv2D(16, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Второй свёрточный слой с функцией активации и слоем пулинга
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Третий свёрточный слой с функцией активации и слоем пулинга
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Четвертый свёрточный слой с функцией активации и слоем пулинга
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Слой распремления с функцией активации
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))

#Полносвязный слой с фенкцией активации
model.add(Dropout(0.5))
model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


datagen = ImageDataGenerator(rescale=1. / 255)
#Обучение
train = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

#Проверка для сверки во время обучения
val = datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

#Тест работы модели
test = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

model.fit(
    train,
    steps_per_epoch=nb_train_samples,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=val,
    validation_steps=nb_validation_samples)

#Вывод результата
scores = model.evaluate_generator(test, nb_test_samples // batch_size)
print("Аккуратность на тестовых данных: %.2f%%" % (scores[1]*100))

#Сохранение модели
model.save('mf.h5')

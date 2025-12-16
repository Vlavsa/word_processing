from posix import mkdir
from warnings import filters

import os
import re
import os
import glob
import patoolib
import shutil
import numpy as np
import matplotlib.pyplot as plt


from itertools import chain
from collections import Counter

from navec import Navec
from keras import utils
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, SpatialDropout1D, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping

from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from razdel import tokenize


current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)


WIN_SIZE = 1000
WIN_STEP = 100 
embedding_dim = 300
max_words = 10000      


navec_path = keras.utils.get_file(
    "navec_hudlit_v1_12B_500K_300d_100q.tar",
    "https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar",
    extract=False
)

navec = Navec.load(navec_path)

# Список писателей по которым моделб будет обучаться
# CLASS_LIST = ["Dostoevsky", "Tolstoy", "Turgenev", "Chekhov", "Lermontov", "Blok", "Pushkin", "Gogol", "Gorky", "Herzen", "Bryusov", "Nekrasov" ]
CLASS_LIST = ["Dostoevsky", "Tolstoy", "Turgenev", "Gorky","Bryusov"]

all_texts = {}

for author in CLASS_LIST:
    all_texts[author] = '' 
    for path in glob.glob('./dataset/prose/{}/*.txt'.format(author)) +  glob.glob('./dataset/poems/{}/*.txt'.format(author)): 
        with open(f'{path}', 'r', errors='ignore') as f: 
            text = f.read()

        all_texts[author]  += ' ' + text.replace('\n', ' ') 
    all_texts[author] = re.sub("[^а-яА-ЯёЁ-]"," ", all_texts[author])
    all_texts[author] = [_.text.lower() for _ in list(tokenize(all_texts[author]))]


# используем генератор цикла для получения длины текстов по каждому автору
total = sum(len(i) for i in all_texts.values())
print(f'Датасет состоит из {total} символов')


print('Общая выборка по писателям:')
for author in CLASS_LIST:
    print(f'{author} - {len(all_texts[author])} символов, доля в общей базе: {len(all_texts[author])/total*100 :.2f}%')



dictionary = list(chain.from_iterable(all_texts.values())) 
dictionary=Counter(dictionary)

sorted_dictionary = dictionary.most_common()


all_words = {}
for i, word in enumerate(sorted_dictionary):
  all_words[word[0]] = i


seq_all_text = {}
for author in CLASS_LIST:
  seq_all_text[author] = [all_words[word] for word in all_texts[author]]


seq_train = list(seq_all_text.values())
seq_train_balance = [seq_train[cls][:40000] for cls in range(len(CLASS_LIST))]


author = "Dostoevsky"
cls = CLASS_LIST.index(author)

print("Фрагмент обучающего текста:")
print("В виде оригинального текста:              ", all_texts[author][:120])
print("Он же в виде последовательности индексов: ", seq_train[cls][:20])


total = sum(len(i) for i in seq_train_balance)
print(f'Датасет состоит из {total} слов')


mean_list = np.array([])
for author in CLASS_LIST:
    cls = CLASS_LIST.index(author)
    print(f'{author} - {len(seq_train_balance[cls])} слов, доля в общей базе: {len(seq_train_balance[cls])/total*100 :.2f}%')
    mean_list = np.append(mean_list, len(seq_train_balance[cls]))

print('Среднее значение слов: ', np.round(mean_list.mean()))
print('Медианное значение слов: ', np.median(mean_list))



def seq_split(sequence, win_size, step):
    return [sequence[i:i + win_size] for i in range(0, len(sequence) - win_size + 1, step)]

def seq_vectorize(
    seq_list,   # Последовательность
    test_split,
    val_split,# Доля на тестовую сборку
    class_list, # Список классов
    win_size,   # Ширина скользящего окна
    step        # Шаг скользящего окна
):

    x_train, y_train, x_test, y_test, x_val, y_val =  [], [], [], [], [], []

    for class_item in class_list:
        cls = class_list.index(class_item)

        first_gate_split = int(len(seq_list[cls]) * ((1-test_split)-val_split))
        second_gate_split = int(len(seq_list[cls]) * (1-val_split))


        vectors_train = seq_split(seq_list[cls][:first_gate_split], win_size, step)
        vectors_test = seq_split(seq_list[cls][first_gate_split:second_gate_split], win_size, step) 
        vectors_val = seq_split(seq_list[cls][second_gate_split:], win_size, step)

        x_train += vectors_train
        x_test += vectors_test
        x_val += vectors_val

        y_train += [keras.utils.to_categorical(cls, len(class_list))] * len(vectors_train)
        y_test += [keras.utils.to_categorical(cls, len(class_list))] * len(vectors_test)
        y_val += [keras.utils.to_categorical(cls, len(class_list))] * len(vectors_val)

    # Возвращаем результатов как numpy-массивов
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test), np.array(x_val), np.array(y_val)

x_train, y_train, x_test, y_test, x_val, y_val = seq_vectorize(seq_train_balance, 0.1, 0.1, CLASS_LIST, WIN_SIZE, WIN_STEP)

print('\n')
print(f'Форма входных данных для обучающей выборки: {x_train.shape}')
print(f'Форма выходных данных (меток) для обучающей выборки: {y_train.shape}')
print('\n')

print(f'Форма входных данных для тестовой выборки: {x_test.shape}')
print(f'Форма выходных данных (меток) для тестовой выборки: {y_test.shape}')
print('\n')

print(f'Форма входных данных для валидационной выборки: {x_val.shape}')
print(f'Форма выходных данных (меток) для валидационной выборки: {y_val.shape}')
print('\n')


# Вывод графиков точности и ошибки
def show_plot(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))
    fig.suptitle('График процесса обучения модели')
    ax1.plot(history.history['accuracy'],
               label='График точности на обучающей выборке')
    ax1.plot(history.history['val_accuracy'],
               label='График точности на проверочной выборке')
    ax1.xaxis.get_major_locator().set_params(integer=True) # На оси х показываем целые числа
    ax1.set_xlabel('Эпоха обучения')
    ax1.set_ylabel('График точности')
    ax1.legend()

    ax2.plot(history.history['loss'],
               label='Ошибка на обучающей выборке')
    ax2.plot(history.history['val_loss'],
               label='Ошибка на проверочной выборке')
    ax2.xaxis.get_major_locator().set_params(integer=True) # На оси х показываем целые числа
    ax2.set_xlabel('Эпоха обучения')
    ax2.set_ylabel('Ошибка')
    ax2.legend()
    plt.show()

# Функция вывода предсказанных значений
def show_confusion_matrix(y_true, y_pred, class_labels):

    cm = confusion_matrix(np.argmax(y_true, axis=1),
                          np.argmax(y_pred, axis=1),
                          normalize='true')
    
    cm = np.around(cm, 3)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f'Матрица ошибок', fontsize=18)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(ax=ax)
    plt.gca().images[-1].colorbar.remove()  # Убираем ненужную цветовую шкалу
    plt.xlabel('Предсказанные классы', fontsize=16)
    plt.ylabel('Верные классы', fontsize=16)
    fig.autofmt_xdate(rotation=45)          # Наклон меток горизонтальной оси
    plt.show()

    print('\nСредняя точность распознавания: {:3.0f}%'.format(100. * cm.diagonal().mean()))


word_index = all_words
embeddings_index = navec

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


def build_model():
  model = Sequential()
  model.add(Embedding(max_words, embedding_dim, input_length=WIN_SIZE, weights=[embedding_matrix]))
  model.add(BatchNormalization())
  model.add(Dense(40, activation="relu"))
  model.add(Dropout(0.6))
  model.add(BatchNormalization())
  model.add(Flatten())
  model.add(Dense(len(CLASS_LIST), activation='softmax'))

  model.layers[0].trainable = False

  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  model.summary()

  return model

model = build_model()

history = model.fit(x_train, y_train, epochs=50, batch_size=64, validation_data=(x_val, y_val))

y_pred = model.predict(x_test)
show_confusion_matrix(y_test, y_pred, CLASS_LIST)
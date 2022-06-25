# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# inicializar a taxa de aprendizado inicial, número de épocas para treinar,
# e tamanho do lote
INIT_LR = 1e-4
EPOCHS = 20
BS = 32


# DIRECTORY = r"C:\Mask Detection\CODE\Face-Mask-Detection-master\dataset"
DIRECTORY = r"C:\Users\User\Desktop\Faculdade\SI\SIA\Algoritmo\dataset"
CATEGORIES = ["with_mask", "without_mask"]

# pegue a lista de imagens em nosso diretório de conjunto de dados e inicialize
# a lista de dados (ou seja, imagens) e imagens de classe
print("[INFO] carregando imagens...")


data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224)) 
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(category)

# execute a codificação one-hot nos rótulos
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.20, stratify=labels, random_state=42)

# construir o gerador de imagem de treinamento para aumento de dados
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# carregar a rede MobileNetV2, garantindo que os conjuntos de camadas FC principais sejam
# left off
baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(224, 224, 3)))

# construir a cabeça do modelo que será colocada em cima do
# o modelo básico
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# coloque o modelo FC da cabeça em cima do modelo base (isso se tornará
# o modelo real que vamos treinar)
model = Model(inputs=baseModel.input, outputs=headModel)

# faz um loop sobre todas as camadas no modelo base e as congela para que elas
# *não* ser atualizado durante o primeiro processo de treinamento
for layer in baseModel.layers:
    layer.trainable = False

# compilar nosso modelo
print("[INFO] compilando modelo...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# treinar o chefe da rede
print("[INFO] cabeça de treinamento...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)

# fazer previsões sobre o conjunto de testes
print("[INFO] avaliando rede...")
predIdxs = model.predict(testX, batch_size=BS)

# para cada imagem no conjunto de teste, precisamos encontrar o índice do
# rótulo com a maior probabilidade prevista correspondente
predIdxs = np.argmax(predIdxs, axis=1)

# mostra um relatório de classificação bem formatado
print(classification_report(testY.argmax(axis=1), predIdxs,
                            target_names=lb.classes_))

# serializar o modelo para o disco
print("[INFO] salvando o modelo do detector de máscara...")
model.save("mask_detector.model", save_format="h5")

# plota a perda e a precisão do treinamento
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")

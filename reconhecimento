import numpy as np
import os
import cv2
import sqlite3
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.layers import GlobalMaxPooling2D, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16, preprocess_input
from scipy.spatial.distance import cosine

# Diretórios
train_dir = '/content/drive/MyDrive/Processamento_Imagens/TrabalhoReconhecimentoFacial/post-processed'
validation_dir = '/content/drive/MyDrive/Processamento_Imagens/TrabalhoReconhecimentoFacial/post-processed'
integrante_grupo_img = '/content/drive/MyDrive/img-proc/imagem_grupo/ismar.jpeg'
integrante_grupo_mask = '/content/drive/MyDrive/img-proc/imagem_grupo/ismar_mascara.jpeg'

# Carregamento e configuração dos dados
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(112, 112),
    batch_size=32,
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(112, 112),
    batch_size=32,
)

# Modelo VGG16
vgg = VGG16(weights='imagenet', include_top=False, input_shape=(112, 112, 3))

for layer in vgg.layers:
    layer.trainable = False

x = GlobalMaxPooling2D()(vgg.output)
x = Dense(512, activation='relu')(x)
output = Dense(2996, activation='softmax')(x)  # Substitua 2996 pelo número de classes

model = Model(inputs=vgg.input, outputs=output)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])

NUM_EPOCHS = 1
BATCH_SIZE = 16

model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

model.save('/content/drive/MyDrive/Processamento_Imagens/Modelos_treinados')

# Funções de extração de características
def extract_feature_vector(img_path, model):
    img = image.load_img(img_path, target_size=(112, 112))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    features = model.predict(img_data)
    return features.flatten()

# Conexão com o banco de dados SQLite
conn = sqlite3.connect('imagens_db.sqlite')
cursor = conn.cursor()

# Função para converter vetor descritor em string
def array_to_string(array):
    return ','.join(map(str, array))

# Criação da tabela no banco de dados, se não existir
cursor.execute('''
    CREATE TABLE IF NOT EXISTS imagens_descritores (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nome_arquivo TEXT,
        vetor_descritor BLOB
    )
''')

# Inserção de imagens e vetores descritores no banco de dados
def insert_into_db(directory):
    for nome_arquivo in os.listdir(directory):
        caminho_imagem = os.path.join(directory, nome_arquivo)
        if nome_arquivo.lower().endswith(('.png', '.jpg', '.jpeg')):
            feature_vector = extract_feature_vector(caminho_imagem, vgg)
            feature_vector_str = array_to_string(feature_vector)
            cursor.execute("INSERT INTO imagens_descritores (nome_arquivo, vetor_descritor) VALUES (?, ?)",
                           (nome_arquivo, feature_vector_str))

insert_into_db('/content/drive/MyDrive/img-proc/post-processed')

# Remoção de registro do integrante do grupo
cursor.execute("DELETE FROM imagens_descritores WHERE nome_arquivo = 'integrante_grupo.jpg';")

# Leitura da imagem do integrante do grupo
imagem_integrante = cv2.imread(integrante_grupo_img)
imagem_integrante = cv2.resize(imagem_integrante, (112, 112))
imagem_integrante = np.expand_dims(imagem_integrante, axis=0)
imagem_integrante = preprocess_input(imagem_integrante)

# Obtenção do vetor de características da imagem do integrante
vetor_descritor = vgg.predict(imagem_integrante)
vetor_descritor_str = array_to_string(vetor_descritor.flatten())

# Inserção do vetor descritor no banco de dados
cursor.execute("INSERT INTO imagens_descritores (nome_arquivo, vetor_descritor) VALUES (?, ?)",
               ('integrante_grupo.jpg', vetor_descritor_str))

conn.commit()
conn.close()

# Leitura da imagem com máscara
imagem_mascara = cv2.imread(integrante_grupo_mask)
imagem_mascara = cv2.resize(imagem_mascara, (112, 112))
imagem_mascara = imagem_mascara / 255.0

# Obtenção do vetor descritor da imagem com máscara
vetor_descritor_mascara = vgg.predict(np.expand_dims(imagem_mascara, axis=0)).flatten()

# Conexão ao banco de dados
conn = sqlite3.connect('imagens_db.sqlite')

# Recuperação dos vetores descritores do banco de dados
resultado = conn.execute('SELECT nome_arquivo, vetor_descritor FROM imagens_descritores')

# Comparação do vetor descritor da imagem com máscara com os do banco de dados
maior_similaridade = -1
pessoa_identificada = None

for row in resultado.fetchall():
    nome_arquivo = row[0]
    vetor_descritor = np.fromstring(row[1], dtype=float, sep=',')

    similaridade = 1 - cosine(vetor_descritor, vetor_descritor_mascara)

    if similaridade > maior_similaridade:
        maior_similaridade = similaridade
        pessoa_identificada = nome_arquivo

conn.commit()
conn.close()

print(f"A pessoa identificada é: {pessoa_identificada}")

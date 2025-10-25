import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import os # DagsHub MLflow için

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

import mlflow  # MLflow
import mlflow.keras  # MLflow
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns 

print("Kütüphaneler yüklendi.")

# --- YENİ BÖLÜM: MLFLOW'U DAGSHUB'A BAĞLA ---
# DagsHub'ın MLflow sunucu adresi
os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/mericsimsek344/dlsignmnistwithdvc.mlflow'
# DagsHub kullanıcı adın
os.environ['MLFLOW_TRACKING_USERNAME'] = 'mericsimsek344'
# DagsHub'dan aldığın Access Token
os.environ['MLFLOW_TRACKING_PASSWORD'] = '90eb5cda0404ab9a05fd674d26e57d372858e1c6'
# -----------------------------------------------

# --- 1. Veri Yükleme ve İşleme ---
try:
    train = pd.read_csv('data/sign_mnist_train.csv')
    test = pd.read_csv('data/sign_mnist_test.csv')
    print("Veri setleri 'data/' klasöründen okundu.")
except FileNotFoundError:
    print("HATA: Veri setleri bulunamadı. (Docker build sırasında dvc pull hatası?)")
    exit()

#
# --- HATA BURADAYDI: EKSİK SATIRLAR ---
# Pandas DataFrame'lerini Numpy dizilerine dönüştür
train_data = np.array(train, dtype = 'float32')
test_data = np.array(test, dtype='float32')
# --- DÜZELTME TAMAMLANDI ---
#

# (Class_names listesi ve diğer ön işleme adımları...)
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y' ]

#Normalize / scale X values
X_train = train_data[:, 1:] /255.
X_test = test_data[:, 1:] /255.

#Convert y to categorical
y_train = train_data[:, 0]
y_train_cat = to_categorical(y_train, num_classes=25)

y_test = test_data[:,0]
y_test_cat = to_categorical(y_test, num_classes=25)

#Reshape for the neural network
X_train = X_train.reshape(X_train.shape[0], *(28, 28, 1))
X_test = X_test.reshape(X_test.shape[0], *(28, 28, 1))
print("Veri ön işleme tamamlandı.")

# --- 2. MLflow Deney Başlangıcı ---
BATCH_SIZE = 128
EPOCHS = 10
DROPOUT_RATE = 0.2

mlflow.set_experiment("Sign_Language_Classifier")
print("MLflow deneyi başlatılıyor...")

with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id}")
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("dropout", DROPOUT_RATE)
    
    mlflow.keras.autolog() 

    # --- 3. Model Oluşturma ve Eğitim ---
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape = (28,28,1), activation='relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(DROPOUT_RATE))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(DROPOUT_RATE))
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(DROPOUT_RATE))
    
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(25, activation = 'softmax'))

    model.compile(loss ='categorical_crossentropy', optimizer='adam', metrics =['acc'])
    
    print("Model eğitimi başlıyor...")
    history = model.fit(X_train, y_train_cat, 
                        batch_size = BATCH_SIZE, 
                        epochs = EPOCHS, 
                        verbose = 1, 
                        validation_data = (X_test, y_test_cat))
    print("Model eğitimi tamamlandı.")
    
    # --- 4. Manuel Loglama (Grafikler ve Ekstra Metrikler) ---
    print("Grafikler oluşturuluyor ve artifact olarak loglanıyor...")
    
    # Not: matplotlib grafikleri için 'Agg' backend'ini kullanmak,
    # sunucu ortamlarında (Docker gibi) 'display' hatasını engeller.
    import matplotlib
    matplotlib.use('Agg')

    # Loss grafiği
    plt.figure()
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig("loss_plot.png")
    mlflow.log_artifact("loss_plot.png")
    plt.close() # plt.show() yerine bu olmalı

    # Accuracy grafiği
    plt.figure()
    plt.plot(history.history['acc'], label='Training acc')
    plt.plot(history.history['val_acc'], label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig("accuracy_plot.png")
    mlflow.log_artifact("accuracy_plot.png")
    plt.close() # plt.show() yerine bu olmalı

    # Final Test Accuracy (Sklearn ile)
    predictions_prob = model.predict(X_test)
    prediction = np.argmax(predictions_prob, axis=1)
    accuracy = accuracy_score(y_test, prediction)
    print(f'Final Test Accuracy Score = {accuracy:.4f}')
    mlflow.log_metric("final_test_accuracy", accuracy)

    # Confusion Matrix
    cm = confusion_matrix(y_test, prediction)
    fig, ax = plt.subplots(figsize=(14, 14))
    sns.heatmap(cm, annot=True, linewidths=.5, ax=ax, fmt='d', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close() # plt.show() yerine bu olmalı

print(f"\nMLflow deneyi tamamlandı. Loglar DagsHub'a gönderildi.")
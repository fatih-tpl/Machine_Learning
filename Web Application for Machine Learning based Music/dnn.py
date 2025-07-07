import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import os
import pickle
import librosa
import librosa.display
import IPython.display as ipd
from IPython.display import Audio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential

# CSV Dosyasını okuma ve df değişkenine atma
df = pd.read_csv("/content/drive/MyDrive/machine_learning/features.csv")
df.head()

df

#df de boş hücre olup olmadığını kontrol eder. Boş hücreleri ekrana bastırır.
print("Columns containing missing values",list(df.columns[df.isnull().any()]))

# Etiketlerin sayısal hale getirilmesi ---> converter.fit_transform

# Blues - 0
# Classical - 1
# Country - 2
# Disco - 3
# Hip-hop - 4
# Jazz - 5
# Metal - 6
# Pop - 7
# Reggae - 8
# Rock - 9

class_encod=df.iloc[:,-1]
converter=LabelEncoder()
y=converter.fit_transform(class_encod)
y

# Eğitim için artık gerekli olmadığı için önce 'filename' sütununun var olup olmadığını kontrol et filename sütununu sil.
if 'filename' in df.columns:
    df = df.drop(labels="filename", axis=1)
else:
    print("DataFrame'de 'filename' sütunu bulunamadı")

from sklearn.preprocessing import StandardScaler
fit=StandardScaler()                                      # Veriyi sıfır ortalama ve birim varyansla ölçeklendirir. Bu, modelin eğitiminde daha hızlı ve verimli sonuçlar elde edilmesine yardımcı olabilir.
X=fit.fit_transform(np.array(df.iloc[:,:-1],dtype=float)) # Veriyi ölçeklendirir ve sonucu X'e atar. X etiketler hariç tüm özellikleri içerir.

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

# Giriş ve gizli nöronlar için en yaygın kullanılan aktivasyon fonksiyonu olan relu'yu kullanırken, çıkış nöronları için softmax aktivasyon fonksiyonunu kullanıyoruz.
model=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(X.shape[1],)),                         
    tf.keras.layers.Dropout(0.2),                                                

    tf.keras.layers.Dense(512,activation='relu'),                               
    keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(256,activation='relu'),                                
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(10,activation='softmax'),                           
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)                       

model.compile(optimizer=optimizer,
             loss="sparse_categorical_crossentropy",                        
              metrics=["accuracy"])

model.summary()

# Model eğitim verisiyle eğitilir.
model_history = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 500, batch_size=256)     # Modeli 500 defa eğitim verisiyle çalıştır.

test_loss,test_acc = model.evaluate(X_test,y_test,batch_size=256)
print("The test loss is ",test_loss)
print("The best accuracy is: ",test_acc*100)

# Eğitim ve test verilerinin nasıl performans gösterdiğini gösterdik.
Validation_plot(model_history)

# Sample testing
sample = X_test
sample = sample[np.newaxis, ...]
prediction = model.predict(X_test)
predicted_index = np.argmax(prediction, axis = 1)
print("Expected Index: {}, Predicted Index: {}".format(y_test, predicted_index))

# Plotting the confusion matrix for analizing the true positives and negatives
import seaborn as sn
import matplotlib.pyplot as plt
pred_x = model.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,predicted_index )
cm

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

labels = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9']

# Görselleştirme
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

X_test.size
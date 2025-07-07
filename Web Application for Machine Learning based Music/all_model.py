# Ortak Veri ve Özellik Çıkarma İşlemleri
import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# Veri Seti Yolu
DATA_PATH = "Data/genres_original"  # GTZAN veri setinin olduğu yol

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y=audio)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        return np.concatenate([
            np.mean(mfccs.T, axis=0),
            np.mean(chroma.T, axis=0),
            np.mean(zcr.T, axis=0),
            np.mean(spectral_contrast.T, axis=0)
        ])
    except Exception as e:
        print(f"Hata: {e} - {file_path}")
        return None

# Özellik ve Etiketleri Hazırlama
features = []
labels = []

for genre in os.listdir(DATA_PATH):
    genre_path = os.path.join(DATA_PATH, genre)
    if os.path.isdir(genre_path):
        for file_name in os.listdir(genre_path):
            file_path = os.path.join(genre_path, file_name)
            feature = extract_features(file_path)
            if feature is not None:
                features.append(feature)
                labels.append(genre)

# DataFrame Oluşturma
df = pd.DataFrame(features)
df['label'] = labels

# Etiketleri Sayısallaştırma
encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['label'])

# Eğitim ve Test Verisine Bölme
X = df.iloc[:, :-1]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veri Normalizasyonu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 1. Random Forest Modeli
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print(f"Random Forest doğruluğu: {rf_accuracy:.2f}")

# 2. SVM Modeli
from sklearn.svm import SVC
svm_model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
svm_model.fit(X_train, y_train)
svm_y_pred = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_y_pred)
print(f"SVM doğruluğu: {svm_accuracy:.2f}")

# 3. KNN Modeli
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
knn_y_pred = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_y_pred)
print(f"KNN doğruluğu (k=3): {knn_accuracy:.2f}")

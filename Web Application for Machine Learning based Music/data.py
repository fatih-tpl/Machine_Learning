# data.py
import os                                          
import librosa                                        
import numpy as np                                    
import pandas as pd                                  
from sklearn.preprocessing import LabelEncoder        

"""Bir ses dosyasindan özellikleri cikarir."""
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)                            #Verilen ses dosyasını yükler
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)                 #Mel-Frekans Kepstral Katsayıları çıkarır (13 adet).  
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)                    #Sesin tonal yapısını (kromagram) çıkarır.
        zcr = librosa.feature.zero_crossing_rate(y=audio)                       #Dalganın sıfırı geçtiği yerlerin oranını çıkarır.
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)   #Frekans bölgelerindeki spektral kontrastı çıkarır.
        return np.concatenate([                                                 #Bu özelliklerin ortalamasını alır ve tek bir vektör olarak döndürür.
            np.mean(mfccs.T, axis=0),
            np.mean(chroma.T, axis=0),
            np.mean(zcr.T, axis=0),
            np.mean(spectral_contrast.T, axis=0)
        ])
    except Exception as e:
        print(f"Hata: {e} - {file_path}")
        return None
    
"""Belirtilen bir klasördeki ses dosyalarindan özellikleri çikarir ve bir CSV dosyasina kaydeder."""
def prepare_dataset(data_path, output_csv="sfeatures.csv"):
    features = []
    labels = []
    for genre in os.listdir(data_path):                            #genre dizinin altındaki müzik çeşitlerini sırasıyla döndürmek için.
        genre_path = os.path.join(data_path, genre)
        if os.path.isdir(genre_path):
            for file_name in os.listdir(genre_path):               #müzik çeşidinde bulunan ses dosyalarını tarar.       
                file_path = os.path.join(genre_path, file_name)
                feature = extract_features(file_path)
                if feature is not None:
                    features.append(feature)
                    labels.append(genre)

    df = pd.DataFrame(features)                                    #features lar için sütun oluşturur.
    df['label'] = labels                                           #sütunlara label ismi verilir.
    df.to_csv(output_csv, index=False)
    print(f"Özellikler {output_csv} dosyasına kaydedildi.")
    return df

"""Kaydedilmiş bir CSV dosyasindan veri setini yükler."""
def load_dataset(csv_file):
    df = pd.read_csv(csv_file)
    encoder = LabelEncoder()
    df['label'] = encoder.fit_transform(df['label'])
    return df, encoder

#prepare_dataset("sDataset\sgenres_original","sfeatures.csv")
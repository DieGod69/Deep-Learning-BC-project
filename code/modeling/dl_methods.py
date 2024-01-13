import os
import warnings
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

warnings.filterwarnings('ignore')


class Dl():
    
    def __init__(self):
        pass
    
    
    def extract_images(self, path_images, model):
        images_list = []

        for path_image in path_images:
            image = load_img(path_image, target_size=(224, 224))
            images = img_to_array(image)
            images = np.expand_dims(images, axis=0)
            images = preprocess_input(images)
            features = model.predict(images, verbose=0)
            images_list.append(features.flatten())
        return np.array(images_list)
        
        
    def image_show(self,path_images, labels, cluster_id):
        cluster_index = np.where(labels == cluster_id)[0]
        sample_indices = np.random.choice(cluster_index, 5, replace=False)
        plt.figure(figsize=(12, 5))
        for i, idx in enumerate(sample_indices):
            plt.subplot(1, 5, i+1)
            image = load_img(path_images[idx], target_size=(224, 224))
            plt.imshow(image)
            plt.axis('off')
        plt.show()
        
    
    def validation(self, confusion):
        accuracy = np.trace(confusion) / np.sum(confusion)
        accuracy = np.trace(confusion) / np.sum(confusion)
        precision = np.diag(confusion) / np.sum(confusion, axis=0)
        recall = np.diag(confusion) / np.sum(confusion, axis=1)
        f1_score = 2 * (precision * recall) / (precision + recall)

        average_precision = np.mean(precision)
        average_recall = np.mean(recall)
        average_f1_score = np.mean(f1_score)

        print(f'Acurácia: {accuracy * 100 :.2f}%')
        print(f'Precisão: {average_precision * 100 :.2f}%')
        print(f'Revocação: {average_recall * 100 :.2f}%')
        print(f'F1-Score: {average_f1_score * 100 :.2f}%')
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def load_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return X_train, y_train, X_test, y_test

def preprocess_data(X_train, y_train, X_test, y_test):
    # Нормализация пикселей [0, 255] -> [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Преобразование в вектор
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    # One-hot encoding для меток
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return X_train, y_train, X_test, y_test

def plot_results(train_losses, train_accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses)
    ax1.set_title('Потери во время обучения')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Потери')
    
    ax2.plot(train_accuracies)
    ax2.set_title('Точность во время обучения')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Точность')
    
    plt.tight_layout()
    plt.show()

def plot_predictions(model, X_test, y_test, num_samples=10):
    indices = np.random.choice(X_test.shape[0], num_samples, replace=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        img = X_test[idx].reshape(28, 28)
        pred = model.predict(X_test[idx:idx+1])[0]
        true = np.argmax(y_test[idx])
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Предсказание: {pred}, Истина: {true}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

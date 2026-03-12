import numpy as np
from mlp import MultilayerPerceptron
from utils import load_mnist, preprocess_data, plot_results, plot_predictions

def main():
    print("Загрузка данных MNIST...")
    X_train, y_train, X_test, y_test = load_mnist()
    
    print("Предобработка данных...")
    X_train, y_train, X_test, y_test = preprocess_data(X_train, y_train, X_test, y_test)
    
    print("Создание модели...")
    model = MultilayerPerceptron(
        input_size=784,
        hidden1_size=256,
        hidden2_size=256,
        output_size=10,
        learning_rate=0.01
    )
    
    print("Обучение модели...")
    losses, accuracies = model.train(X_train, y_train, epochs=50, batch_size=32)
    
    print("Тестирование модели...")
    test_accuracy = model.accuracy(X_test, y_test)
    print(f"Точность на тестовых данных: {test_accuracy:.4f}")
    
    print("Визуализация результатов...")
    plot_results(losses, accuracies)
    plot_predictions(model, X_test, y_test)

if __name__ == "__main__":
    main()

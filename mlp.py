import numpy as np

class MultilayerPerceptron:
    def __init__(self, input_size=784, hidden1_size=256, hidden2_size=256, output_size=10, learning_rate=0.01):
        # Инициализация весов Xavier
        self.W1 = np.random.randn(input_size, hidden1_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden1_size))
        
        self.W2 = np.random.randn(hidden1_size, hidden2_size) * np.sqrt(2.0 / hidden1_size)
        self.b2 = np.zeros((1, hidden2_size))
        
        self.W3 = np.random.randn(hidden2_size, output_size) * np.sqrt(2.0 / hidden2_size)
        self.b3 = np.zeros((1, output_size))
        
        self.learning_rate = learning_rate
        
    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)
        
    def leaky_relu_derivative(self, x, alpha=0.01):
        return np.where(x > 0, 1, alpha)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        
    def sigmoid_derivative(self, x):
        return x * (1 - x)
        
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.leaky_relu(self.z1)
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.leaky_relu(self.z2)
        
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.sigmoid(self.z3)
        
        return self.a3
        
    def backward(self, X, y, output):
        m = X.shape[0]
        
        # Выходной слой
        dZ3 = output - y
        dW3 = np.dot(self.a2.T, dZ3) / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m
        
        # Второй скрытый слой
        dZ2 = np.dot(dZ3, self.W3.T) * self.leaky_relu_derivative(self.z2)
        dW2 = np.dot(self.a1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        # Первый скрытый слой
        dZ1 = np.dot(dZ2, self.W2.T) * self.leaky_relu_derivative(self.z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # Обновление весов
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        
    def train(self, X, y, epochs=100, batch_size=32):
        losses = []
        accuracies = []
        
        for epoch in range(epochs):
            # Перемешивание данных
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Прямое распространение
                output = self.forward(X_batch)
                
                # Вычисление потерь
                loss = -np.mean(y_batch * np.log(output + 1e-8))
                epoch_loss += loss
                
                # Обратное распространение
                self.backward(X_batch, y_batch, output)
            
            # Метрики
            avg_loss = epoch_loss / (X.shape[0] // batch_size)
            accuracy = self.accuracy(X, y)
            
            losses.append(avg_loss)
            accuracies.append(accuracy)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
                
        return losses, accuracies
        
    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)
        
    def accuracy(self, X, y):
        predictions = self.predict(X)
        y_true = np.argmax(y, axis=1)
        return np.mean(predictions == y_true)

""" 
学习曲线是一种可视化工具，用于展示机器学习模型在训练过程中的表现。
通常，学习曲线以训练集的样本数量或迭代次数为横轴，以模型的准确率为纵轴。
通过绘制学习曲线，我们可以观察到模型在不同训练阶段的表现，从而评估模型的性能
"""

# Generate synthetic data
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
x = np.random.uniform(-3.0, 3.0, size=100)
X = x.reshape(-1, 1)
y = 0.5 * x**2 + x + 2 +np.random.normal(0, 1, 100) #增加噪音

# plt.scatter(X, y)
# plt.show()

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Define a simple neural network
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Linear(1, 1)
    
    def forward(self,x):
        return self.fc(x)

# Instantiate the model, loss function, and optimizer
model = Model()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Define function for tranning the model
def train_model(model,criterion,optimizer,X,y,epochs=100,batch_size=16,validation_split=0.2):
    train_loss_history = []
    val_loss_history = []
    n_samples = X.size(0)
    n_val = int(validation_split * n_samples)
    n_train = n_samples - n_val

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_samples)
        for i in range(0, n_train, batch_size):
            indices = perm[i:i+batch_size]
            batch_X, batch_y = X[indices], y[indices]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        train_loss = criterion(model(X[:n_train]),y[:n_train]).item()
        val_loss = criterion(model(X[n_train:]),y[n_train:]).item()

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)


        if epoch % 10 ==0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    return train_loss_history, val_loss_history


# Train the model

train_losses, val_losses = train_model(model, criterion, optimizer, X_tensor, y_tensor)

# Plot the learning curve
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.legend()
plt.show()






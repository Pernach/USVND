import torch
import torch.nn as nn

class MultiTargetSimpleNN(nn.Module):
    """Простая нейронная сеть
    
    Keyword arguments:
    input_size -- количество признаков
    Return: результат классификации
    """

    def __init__(self, input_size):
        super(MultiTargetSimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)  # 3 выходных нейрона для 3 бинарных классов
        self.sigmoid = nn.Sigmoid()  # Используем сигмоиду для бинарной классификации

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x
    

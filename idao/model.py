import torch.nn as nn


class SkyNet(nn.Module):
    """
        Наш бэйзлайн для счёта 997.99 на Public Test.
    """

    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
                                        nn.Conv2d(1, 6, 5),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2, 2),
                                        nn.Conv2d(6, 10, 5),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2, 2),
                                        nn.BatchNorm2d(10),
                                        nn.Conv2d(10, 8, 5),
                                        nn.ReLU(),
                                        nn.MaxPool2d(8, 8),
                                        nn.Flatten()
        )

        self.linear_combination = nn.Sequential(
                                        nn.Linear(6 * 6 * 8, 64),
                                        nn.ReLU(),
                                        nn.Dropout(0.3),
                                        nn.Linear(64, 32),
                                        nn.Dropout(0.3),
                                        nn.ReLU()
        )
        self.classification = nn.Linear(32, 2)
        self.regression = nn.Linear(32, 6)


    def forward(self, x):
        """
            Функция определяет последовательность производимых преобразований
        при forward pass. Полученный список нужно подать в softmax функцию
        для получения вероятностей.

            Args:
                x - input image (batch_size, 224, 224)
            Returns:
                [
                    (batch_size, 2) - линейная комбинация параметров для классификации
                    (batch_size, 6) - линейная комбинация параметров для регрессии
                    ]        
        """
        x = self.feature_extractor(x)
        x = self.linear_combination(x)
        clas = self.classification(x)
        regr = self.regression(x)
        return clas, regr


if __name__ == "__main__":
    net = Net()
    print(net)


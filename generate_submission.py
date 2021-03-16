import torch
import torchvision
import pandas as pd

from idao.model import SkyNet
from idao.datasets import ImageFolderWithPaths



class ContestSubmitter:

    """
        Класс, настроенный на отправку решения по второму треку.
    Подгружает наш `SkyNet`, используя параметры ниже. 

    `BATCH_SIZE`  - количество одновременно предсказываемых изображений.
    `IMAGES_PATH` - путь до изображений на яндекс контесте
    `WEIGHTS`     - веса для SkyNet

    """

    BATCH_SIZE = 150
    IMAGES_PATH = 'tests/'
    WEIGHTS = 'weights/skynet_997_99.pth'

    def __init__(self):
        
        # если на контестере есть видеокарта, то подрубаем её
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # подгружаем используемую модель        
        self.model = self.compile_model().eval()


    def compile_model(self):
        """
            Функция иницализирует обученную модель и подгружает к ней веса.
        """
        model = SkyNet()
        weights = torch.load(self.WEIGHTS, map_location=torch.device(self.device))
        model.load_state_dict(weights)
        return model

        
    def prepare_submission(self):
        """
            Функция описывает главную логику этого класса.
        Сначала мы подгружаем аугментатор, который будет применяться 
        для каждой фотографии в генераторе изображений. После аугментатора
        подсоединяем генератор изображений и, наконец, запускаем процесс 
        предсказания
        """

        data_transforms = self.get_augmentation()
        dataloader = self.get_dataloader(data_transforms)
        self.predict(dataloader)


    def get_augmentation(self):
        """
            Функция описывает, какие нужно применить к каждой
        картинке преобразования, перед тем, как запустить их
        в нейронку
        """
        return torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(),
            torchvision.transforms.CenterCrop(224),     
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485],
                                             [0.229])
        ])
         

    def get_dataloader(self, data_transforms):
        """
            Даталоадеры на языке пайторч - генераторы, которые
        находят изображения в указанных аудиториях. При итерации
        через эти генераторы мы получаем батчи подготовленных
        для предсказания изображений.
        """

        dataset = ImageFolderWithPaths(self.IMAGES_PATH, data_transforms)
        return torch.utils.data.DataLoader(dataset, batch_size=self.BATCH_SIZE)
         
    
    def predict(self, dataloader):
        """
            Основная функция для создания предсказаний. Полностью описывает логику
        потоков входных данных из даталоадера в список словарей, где каждый словарь
        характеризует предсказание для конкретного изображения. Полученный список
        словарей затем преобразовывается в `pandas.DataFrame` для сохранения в .csv
        """

        data = []
        # сопоставление энергии и предсказания модели-регрессора
        classes_to_idx = {'1': 0, '10': 1, '20': 2, '3': 3, '30': 4, '6': 5}
        idx_to_classes = {value: key for key, value in classes_to_idx.items()}
        
        # указываем торчу, что не нужно считать градиенты
        with torch.no_grad():
            # из генератора достаем только изображения и их полные адреса
            for inputs, _, paths in dataloader:
                
                # переносим матрицы на доступный девайс (гпу или цпу)
                inputs = inputs.to(self.device)

                # модель одновременно делает классификацию и регрессию
                output_cl, outputs_pred = self.model(inputs)

                # вероятность каждого из классов для классификатора
                prob_cl = torch.nn.Softmax(dim=1)(output_cl)[:, 1].numpy()

                # предсказанные классы регрессором
                prob_reg = torch.max(torch.nn.Softmax(dim=1)(outputs_pred), 1).indices
                # переводим классы в значения энергии
                prob_reg = [idx_to_classes[i] for i in prob_reg.numpy()]

                # формируем список словарей, которые превратятся в csv
                data.extend([{
                    'id': link.split('/')[-1].split('.png')[0],
                    'classification_predictions': 1 - p_cl,
                    'regression_predictions': p_reg,
                } for (link, p_cl, p_reg) in zip(paths, prob_cl, prob_reg)])
            
        # превращаем список словарей в .csv
        pd.DataFrame(data).set_index('id').to_csv('submission.csv')
    


if __name__ == "__main__":
    submitter = ContestSubmitter()
    submitter.prepare_submission()


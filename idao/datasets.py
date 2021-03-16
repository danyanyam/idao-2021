import torchvision


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    """
        Кастомная модификация датасета от `torchvision`, направленная
    на то, чтобы генератор изображений кроме самих матриц картинок
    и лэйблов классов для них, возвращал еще и путь до подаваемого
    в батч изображения.
    """

    def __getitem__(self, index):
        """
            Модифицируем только способ извлечения данных, которым
        пользуется `torch.utils.data.DataLoader` при генерации
        изображений.
        """
        # оставляем стандартное поведение
        original_tuple = super().__getitem__(index)

        # извлекаем путь до фото из внутренних переменных
        path = self.imgs[index][0]

        # добавляем путь к тому, что нам до этого генерировал класс
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
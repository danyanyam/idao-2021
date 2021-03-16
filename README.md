# IDAO 2021 отборочные

Для воспроизведения наших результатов:

Создаем виртуальное окружение и активируем его: 
```bash
python3 -m venv venv && source venv/bin/activate 
```

Обновляем менеджер библиотек и загружаем зависимости из файла `requirements.txt`
```bash
pip3 install --upgrade pip && pip3 install -r requirements.txt
```

Архивируем результаты для отправки в контест:
```bash
. ./zip.sh
````
# ml_project Artem Kugushev
Проект имеет модульную структуру. 

+ `models` - содержит файл с функциями обучения, предсказания, создания пайплайна, а так же сохраненные модели и резульаты прогона на валидационном множестве (для каждой модели отдельная директория)
+ `notebooks` - соджержит ноутбук с EDA
+ `logs` - содержит файл с функцией создания логгера (необходимость отдельного модуля связана с использованием логгера в разных пайплайнах и тестах), так же сюда пишутся логи в файл full_logs.log (он игнорируется в git)
+ `data` - содержит датасет для обучения и директорию *dataclass* с датаклассами
+ `tests` - содержит файлы с тестами для отдельных функций в train_pipeline и predict_pipeline
+ `configs` - содержит конфиги в виде yaml для обучения (/train/) для 2х моделей классификации (LogReg, DecisionTree); конфиг логгера - можно выбрать в других конфигах тип логгера(stream или file) в поле log_format; predict config, в котором выбирается модель, по которой делаются предсказания; небольшие конфиги для тестов

В базовой директории содержатся файлы *train_pipeline* и *predict_pipeline*, которые соответственно запускают обучение и предсказание модели(если такая существует); *requirements.txt*, и gitignore файл.
В каждой директории прописан файл `__init__.py`.

Так же настроен ci с помощью yaml конфига, с помощью которого при push/PR запускаются функции тестирования (train и predict).


## Команды для запуска основных частей проекта из командной строки

**Train pipeline** `python3 train_pipeline.py ./configs/train/{tree/log_reg}_train_config.yaml`

**Predict pipeline** `python3 predict_pipeline.py ./configs/predict_config.yaml`

**Tests** `python3 -m unittest ./tests/{predict/train}_pipeline_test.py`

## HW2

*Build from online_reference/* `docker build -t artemkgushev/homework2 .`
*Pull from [dockerHub](https://hub.docker.com/repository/docker/artemkgushev/homework2)* `docker pull artemkgushev/homework2`
*Run* `docker run -d -p 8888:8888 artemkgushev/homework2`

*Run request in docker terminal* `python3 -m request`
*Run test from docker terminal* `python3 -m pytest test.py`

Работоспособность predict проверялась с помощью Postman.

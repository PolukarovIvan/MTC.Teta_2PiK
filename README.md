## Прогнозирование группы пользователей для осуществления таргетированной рекламы банковского продукта.
### Команда __2𝝅k__: 
*  Полукаров Иван
*  Портнов Роман
*  Красильников Денис

### Описание репозитория:
* models\ &mdash; папка с итоговыми моделями
* BPL.csv &mdash; исходный набор данных
* Personal_Loan_project.ipynb &mdash; ноутбук с решением
* data_engeneering_pipeline.pkl &mdash; пайплайн предобработки данных
* dtc_raw_tree.dot &mdash; служебный файл для визуализации решающего дерева
* dtc_raw_tree.png &mdash; изображение решающего дерева
* scoring_df.csv &mdash; метрики всех моеделей
 

### Описание проекта
Задача банка предложить потребительский кредит и при этом максимизировать [показатель конверсии](https://www.unisender.com/ru/support/about/glossary/chto-takoe-cr-conversion-rate/), который равен отношению количества пользователей, принявших кредит, к общему количеству пользователей *(precision)*. Решение этой задачи актуально для банка, потому что это позволит сократить расходы на персональный маркетинг, но при этом достичь максимальной выгоды. Данный проект оптимизирует выбор кандидатов для персонального предложения о предоставлении кредита. В процесс отбора кандидатов можно внедрить алгоритм отсева потенциальных кандидатов, готовых взять кредит.  

### Постановка бизнес и математической задачи
Предположительно, нам известно сколько было затрачено времени и средств на проведение прошлогодней кампании. Для измерения бизнес цели мы планируем использовать сравнение затраченных и полученных средств текущей и прошлогодней кампании. Критерий успеха — доход, превышающий доход прошлогодней кампании (или превышение предсказанных показателей дохода, тк может быть тренд увеличения прибыли).

С математической точки зрения мы хотим, чтобы как можно больше людей из тех, кого мы выберем приняли наше предложение. Нам требуется построить бинарный классификатор, с вероятностными прогнозами. Для повышения показателя конверсии нам необходимо оптимизировать precision, но так же необходимо охватить больше потенциально согласных пользователей, поэтому мы будем оптимизировать __F1 score__ и следить за __precision__.

### Данные
Dataset был взят с платформы [kaggle](https://www.kaggle.com/krantiswalke/bank-personal-loan-modelling/code). 

___Пример данных___
[![data](https://i.imgur.com/D3IqggG.png)](https://www.kaggle.com/krantiswalke/bank-personal-loan-modelling)

### Валидация данных и оценка потенциала
В данных отсутвуют пропуски, достаточно информации о клиенте и взаимоотношений с банком. Размер датасета 5000 строк.  
В качестве baseline было принято решение разделить людей по зарплате на две группы.
Результаты: 
* f1 score: 0.4969
* precision score: 0.3496

Данные результаты говорят о возможном повышении показателя конверсии до 35.0%

### Оценка экономического эффекта
Предварительная оценка внедрения нашего проекта позволит повысить показатель конверсии с 9,2% до ~89.7% _(по результатам precision на [тестовой выборке](https://github.com/deethereal/MTC.Teta_2PiK/blob/master/Personal_Loan_project.ipynb))_, что позволит сэкономить на персональном маркетинге, при этом получить максимальную прибыль. Приведены результаты работы 	DecisionTreeClassifier, работу которого можно впоследствии легко интерпретировать. Для контроля количества верно предсказнных результатов посчитаем __f1 score__: 0,8715. Результаты метрик нас устраивают и на этом можно завершить построение модели и оценить экономический эффект. 
Пусть затраты на маркетинг на одного клиента составляют __N__, средний доход с одного клиента __M__, общее количеств клиентов __n__. Количество новых кандидатов относительно старого способа составляет 8.7%
Тогда расчет экономического эффекта будет следующим
```math
Profit (old) = (M * 0.092 - N) * n
Profit (new) = (M * 0.897 - N) * 0.087 * n
Delta = (-0.013 * M + 0.913 * N) * n
```
Точную оценку эффекта мы дать не можем, так как нам не доступны данные. Но мы можем предположить, что изначально было 5000 клиентов, на привлечение клиента мы тратим 50 рублей, а средняя потенциальная выручка с клиента составляет 1000 рублей. 

Вычисляя по формуле дельты мы получаем, что при внедрении нашей системы доход банка равен 163250 рублей. Таким образом внедрение нашей модели положительно сказывает на доходах банка.
На данный момент с увеличением качества модели растет и показатель конверсии, и recall. Следовательно выручка увеличивается. Для сравнения возьмем RandomForestClassifier с f1 score 0.8927 и precision 0.9294. Тогда выручка составит 
```math
Delta2 = (-0.013 * M + 0.915 * N) * n = (-0.013 * 1000 + 0.915 * 50) * 5000 = 163750 руб
```
С ростом качеста модели на 2 процента выручка выросла на 1 процент. Предположительно, что с большим ростом качества модели доход вырастет слабо.  


### Анализ DecisionTree

[![tree](https://i.imgur.com/Z5gH5oo.png)](https://i.imgur.com/Z5gH5oo.png)

Визуальный анализ показывает, что больше всего на результат влияет доход людей. Данный граф позволяет самому понять, почему алгоритм сделал такой выбор, посмотрев на характеристики конкретного человека. 

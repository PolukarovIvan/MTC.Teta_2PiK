## Прогнозирование группы пользователей для осуществления таргетированной рекламы банковского продукта
### Команда __2𝝅k__: 
*  Полукаров Иван
*  Портнов Роман
*  Красильников Денис

### Описание проекта
Задача банка предложить потребительский кредит и при этом максимизировать [показатель конверсии](https://www.unisender.com/ru/support/about/glossary/chto-takoe-cr-conversion-rate/), который равен отношению количества пользователей, принявших кредит, к общему количеству пользователей, которым поступили предложения *(precision)*. Решение этой задачи актуально для банка, потому что это позволит сократить расходы на персональный маркетинг, но при этом достичь максимальной выгоды. Данный проект оптимизирует выбор кандидатов для персонального предложения о предоставлении кредита. В процесс отбора кандидатов можно внерить алгоритм отсева поенциальных кандидатов, готовых взять кредит.  

### Постановка бизнес и математической задачи
Предположительно, нам известно сколько было затрачено времени и средств на проведение прошлогодней кампании. Для измерения бизнес цели мы планируем использовать сравнение затраченных и полученных средств текущей и прошлогодней кампании. Критерий успеха — доход, превышающий доход прошлогодней кампании (или превышение предсказанных показателей дохода, тк может быть тренд увеличения прибыли).

С математической точки зрения мы хотим, чтобы как можно больше людей из тех, кого мы выберем приняли наше предложение. Нам требуется построить бинарный классификатор, с вероятностными прогнозами. Для повышения показателя конверсии нам необходимо оптимизировать precision, но так же необходимо охватить больше потенциально согласных пользователей, поэтому мы будем оптимизировать __F1 score__ и следить за __precision__.

### Данные
Dataset был взят с платформы [kaggle](https://www.kaggle.com/krantiswalke/bank-personal-loan-modelling). 

___Пример данных___
[![data](https://i.imgur.com/D3IqggG.png)](https://www.kaggle.com/krantiswalke/bank-personal-loan-modelling)

### Валидация данных и оценка потенциала
В данных отсутвуют пропуски, достаточно информации о клиенте и взаимоотношений с банком. Размер датасета 5000 строк.  
В качестве baseline было принято решение построить логистическую регрессию.
> [notebook](https://github.com/pam4ek/MTC.Teta_2PiK/blob/master/personal_loan_baseline.ipynb) 
> f1 score: 0.6743
> precision score: 0.7867

Данные результаты говорят о возможном повышении показателя конверсии до 78.7%

### Оценка экономического эффекта
Предварительная оценка внедрения нашего проекта позволит повысить показатель конверсии с 9,6% до ~97.6% _(по результатам precision на [тестовой выборке](https://github.com/deethereal/MTC.Teta_2PiK/blob/master/PipelineAndValidation_format.ipynb))_, что позволит сэкономить на персональном маркетинге, при этом получить максимальную прибыль. Приведены результаты работы 	DecisionTreeClassifier, работу которого можно впоследствии легко интерпретировать. Для контроля количества верно предсказнных результатов посчитаем __f1 score__: 0,884. Результаты метрик нас устраивают и на этом можно завершить построение модели. 
На данный момент с увеличением качества модели падает показатель конверсии, но recall начинает расти. Следовательно выручка увеличивается???
- Опишите логику оценки и приведите расчет;
- Как оценка эффекта будет меняться в зависимости от качества модели?
- Можете ли вы оценить, насколько изменится эффект от роста качества модели на 1%? На 10%?

### Анализ DecisionTree
возможно



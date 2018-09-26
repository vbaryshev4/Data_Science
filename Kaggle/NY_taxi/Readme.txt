https://www.kaggle.com/c/new-york-city-taxi-fare-prediction

Условия:
	In this playground competition, hosted in partnership with Google Cloud and Coursera, you are tasked with predicting the fare amount (inclusive of tolls) for a taxi ride in New York City given the pickup and dropoff locations. While you can get a basic estimate based on just the distance between the two points, this will result in an RMSE of $5-$8, depending on the model used (see the starter code for an example of this approach in Kernels). Your challenge is to do better than this using Machine Learning techniques!

	To learn how to handle large datasets with ease and solve this problem using TensorFlow, consider taking the Machine Learning with TensorFlow on Google Cloud Platform specialization on Coursera -- the taxi fare problem is one of several real-world problems that are used as case studies in the series of courses. To make this easier, head to Coursera.org/NEXTextended to claim this specialization for free for the first month!

Предыстория:
	Член команды застрял в прогрессе с скором ≈ 3.1 Попросил помочь. Датасэт с отложенной выборкой естесвенно без целевой фичи. Это затрудняет количество попыток валидации модели. Таккак количество коммитов вдень ограничено.

Решение:
	1)Вырабоатал гипотезы. Изучил бизнес условия, сорбрал данные как работает такси в Нью-Йорке.

	2) Нашел датасэты за 10 лет и выкачал их ≈ 200ГБ. Это позволило обогатить трэйн и тест выборку. А самое главное теперь мой партнер по команде может не коммить версии маодлей, посколько у нас есть целевой показатель для отложенной выборки.
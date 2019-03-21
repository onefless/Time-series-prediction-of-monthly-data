# Time-series-prediction-of-monthly-data
Provides the code to implement additive and multiplicative decompositoin model to monthly time series for forecast, testing and visualisation. 

#### [Tutorial](https://github.com/onefless/Time-series-prediction-of-monthly-data/blob/master/Tutorial.ipynb) 
Walkthough with Australian tourists data.

Original data
![Original](https://github.com/onefless/Time-series-prediction-of-monthly-data/blob/master/Plots/Original.png?raw=true)

Decomposition components
![components](https://github.com/onefless/Time-series-prediction-of-monthly-data/blob/master/Plots/multi%20%20components.png?raw=true)

Final prediction
![prediction](https://github.com/onefless/Time-series-prediction-of-monthly-data/blob/master/Plots/Prediction.png?raw=true)

#### Download [Module](https://github.com/onefless/Time-series-prediction-of-monthly-data/blob/master/decomposition.py)
Sample demonstration:

	import decomposition
	
	mdl_additive = decomposition.decomposition_additive()
	mdl_multiplicative = decomposition.decomposition_multiplicative()

	# Training and testing
	mdl_additive.fit(train)
	test_pred = mdl_additive.test(test)

	# Prediction 
	prediction = mdl_additive.predict(months = 36)

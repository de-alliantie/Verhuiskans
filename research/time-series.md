# Time series

## Autoregressive (AR)
Maakt de aanname dat de huidige stap in de time series zich laat voorspellen op basis van voorgaande waarden. Kijkt naar de correlatie tussen de waarde die voorspelt wordt en de andere waarden (lag-values) in de dataset. Hoe hoger de correlatie tussen die twee, deste meer gewicht er aan de waarde gegeven wordt voor de voorspelling van de volgende tijdsstap.

Wanneer er geen of lage correlatie is tussen de lag-values en de voorspelde waarde, is dat een indicator dat de time-series niet voorspeld kan worden met de huidige lag-values.

Voorbeeld van [python implementatie](https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/)

## Moving Average (MA)
Moving average neemt een gemiddelde van de vorige tijdstappen om daarme de volgende tijdstap te voorspellen. Nuttig om ruis weg te filteren en werkt aardig op korte tijdsintervallen. Negeert echter lange-termijn trends en is meestal te laat wanneer snelle veranderingen in het signaal optreden. Window size is een parameter die ingesteld moet worden, alsmede het type averaging. 

## Autoregressive Integrated Moving Average (ARIMA)
Wat ARMA lastig maakt is dat je een stationary time-series nodig hebt, waarbij de mean en variance constant blijven over tijd. ARIMA lost dit op door het verschil in de data tussen twee tijdspunten van elkaar af te trekken. Hierdoor word bijvoorbeeld een shifting mean opgelost.

## Seasonal Autoregressive Integrated Moving Average (SARIMA)
Een aanpassing van ARIMA, waarbij seasonality ook wordt meegenomen.

## Conclusie
Klassieke time-series hebben een variabele die veel veranderd over de tijd heen. Bij ons blijft de target variabele juist constant over de tijd tot het verhuis event. Daarom de vraag hoe nuttig klassieke time-series forecasting algoritmes (zoals Autoregressive, Moving average, ARIMA etc.) zijn. Klassieke Time series forecasting is voornamelijk een regression taak, terwijl wij meer een soort binary uitkomst voorspellen.

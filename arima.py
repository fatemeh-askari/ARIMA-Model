def arima_model(data):
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.arima.model import ARIMAResults
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings("ignore")

    frequency = str(input("Enter the frequency of the data (T, H, D, W, M , Y): "))
    index_freq = frequency.upper()
    index = pd.date_range(start=data.index[0], periods=len(data), freq=index_freq)

    d = int(input("Enter the integrated component according to your data (0,1,...): "))
    p = None
    q = None
    best_aic = np.inf
    for i in range(1, 10):
        for j in range(1, 10):
            model = sm.tsa.ARIMA(data, order=(i, d, j), freq=frequency)
            results = model.fit()
            aic = results.aic
            if aic < best_aic:
                best_aic = aic
                p = i
                q = j
    model = ARIMA(data, order=(p, d, q), freq=frequency).fit()

    print_result = input("Do you want the model summary to be shown? (y/n): ")
    if print_result.lower() == 'y':
        print(f"Best p: {p}, Best q: {q}")
        print(model.summary())
    else:
        pass

    print_result = input("Want to show the plot? (y/n): ")
    if print_result.lower() == 'y':
        predictions = model.predict(start=data.index[0], end=data.index[-1], typ='levels')
        print(predictions)
        plt.plot(index, data.values, label='Actual')
        plt.plot(index, predictions.values, label='Predicted')
        plt.legend()
        plt.show()
        
        residuals = model.resid
        plt.plot(residuals)
        plt.title('Residuals Plot')
        plt.show()
    else:
        pass

    n_periods = int(input("Enter the number of periods you want to predict: "))
    model = ARIMA(data, order=(p, d, q), freq=frequency)
    results = model.fit()
    forecast = results.forecast(steps=n_periods)
    print(forecast)

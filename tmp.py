from EuroOptionFFTPricing import EuroOptionFFT
from EuroOptionAnalysisPricing import EuroOption
import numpy as np
import matplotlib.pyplot as plt



if __name__ == '__main__':
    params = {"s": 100.0,  # index level
              "k": 90.0,  # option strike
              "t": 1.0,  # maturity date
              "r": 0.05,  # risk-less short rate
              "sigma": 0.3,  # volatility
              "q": 0,
              "eta": 0.15,
              "alpha": 1.5,
              "N": np.power(2, 10)}

    eurofft = EuroOptionFFT(**params)

    km, p_fft = eurofft.fft_main()
    p_aly = []
    for k in km:
        euro = EuroOption(t=1, k=k, s=100, sigma=0.3, r=0.05, q=0)
        p_aly.append(euro.formulate())
    p_aly = np.array(p_aly)
    error = p_fft - p_aly
    error_pct = error / p_aly
    plt.plot(error_pct[:100])
    plt.show()
    # print(np.interp(90, km, p_fft))



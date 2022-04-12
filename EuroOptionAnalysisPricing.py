#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zhh
@contact:zhangyi2k15@qq.com
@version: 3.8.0
@license: Apache Licence
@file: EuroOptionAnalysisPricing.py
@time: 2022/3/30 13:35
@desc: pricing euro option by finite difference method. 这里；利用显式差分法
"""
import numpy as np
from scipy.stats import norm


class EuroOption(object):
    def __init__(self, t, k, s, sigma, r, q):
        self.t = t
        self.k = k
        self.s = s
        self.sigma = sigma
        self.r = r
        self.q = q

    def formulate(self):
        d_1 = (np.log(self.s / self.k) + (self.r - self.q + np.power(self.sigma, 2) / 2) * self.t) / (
                self.sigma * np.sqrt(self.t))

        d_2 = d_1 - self.sigma * np.sqrt(self.t)
        euro_call = self.s * np.exp(-self.q * self.t) * norm.cdf(d_1) - self.k * np.exp(-self.r * self.t) * norm.cdf(
            d_2)
        return euro_call

    def pay_off(self, S):
        return max(0, S - self.k)


if __name__ == '__main__':
    s = 100.0  # index level
    K = 90.0  # option strike
    T = 1.0  # maturity date
    r = 0.05  # risk-less short rate
    sigma = 0.3  # volatility
    q = 0
    euro = EuroOption(s=s, k=K, t=T, sigma=sigma, r=r, q=0)
    print(euro.formulate())


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zhh
@contact:zhangyi2k15@qq.com
@version: 3.8.0
@license: Apache Licence
@file: EuroOptionFiniteDiffPricing.py
@time: 2022/4/11 9:13
@desc: 
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

    # def formulate(self):
    #     d_1 = (np.log(self.k / self.s) + (self.r - self.q + np.power(self.sigma, 2) / 2) * self.t) / (
    #             self.sigma * np.sqrt(self.t))
    #     d_2 = d_1 - self.sigma * np.sqrt(self.t)
    #     euro_call = self.s * np.exp(-self.q * self.t) * norm.cdf(d_1) - self.k * np.exp(-self.r * self.t) * norm.cdf(
    #         d_2)
    #     return euro_call

    def get_convergence_dx(self, d_t):
        tmp_val = np.abs(self.r - self.q - np.power(self.sigma, 2) / 2)
        up_ = np.power(self.sigma, 2) / tmp_val
        if tmp_val > 0:
            for dx in np.linspace(1e-5, up_, 100):
                if np.power(self.sigma, 2) * d_t / np.power(dx, 2) <= 1:
                    return dx

        return np.sqrt(np.power(self.sigma, 2) * d_t * 2)  # default alpha=1/2

    def finite_diff(self):
        N, M = 100, 100
        dt = self.t / N
        dx = self.get_convergence_dx(dt)
        V = np.zeros((M, N))
        # s_max, s_min = self.k + 20, self.k - 20
        stock_val = []
        for m in range(M):
            stock_val.append(np.exp(m * dx))
            V[m, N - 1] = self.pay_off(np.exp(m * dx))  # 最后时刻的收益

        w = np.power(self.sigma, 2) * dt / np.power(dx, 2)
        tmp = self.r - self.q - np.power(self.sigma, 2) / 2
        for n in range(N - 2, -1, -1):
            for m in range(M - 2, -1, -1):
                V[m, n] = 1 / (1 + self.r * dt) * (
                        (1 - w) * V[m, n + 1] +
                        (w / 2 + np.sqrt(w) / (2 * self.sigma) * tmp * np.sqrt(dt)) * V[m + 1, n + 1] +
                        (w / 2 - np.sqrt(w) / (2 * self.sigma) * tmp * np.sqrt(dt)) * V[m - 1, n + 1]
                )
        return V, np.array(stock_val)


if __name__ == '__main__':
    s = 120
    p = {"t": 1, "k": 100, "s": s, "sigma": 0.3, "r": 0.03, "q": 0}
    eo = EuroOption(**p)
    print(eo.formulate())
    diff, stock = eo.finite_diff()
    for i in range(0, len(stock)-1):
        if stock[i] <= s <= stock[i + 1]:
            print(diff[i, 0])
            break
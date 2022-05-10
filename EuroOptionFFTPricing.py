#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zhh
@contact:zhangyi2k15@qq.com
@version: 3.8.0
@license: Apache Licence
@file: EuroOptionFFTPricing.py
@time: 2022/4/11 9:12
@desc: 
"""

import numpy as np


def SimpsonW(N, eta):
    delt = np.zeros(N, dtype=np.float)
    delt[0] = 1
    j = np.arange(1, N + 1, 1)
    SimpsonW = eta * (3 + (-1) ** j - delt) / 3
    return SimpsonW


class EuroOptionFFT(object):
    def __init__(self, t, k, s, sigma, r, q, eta, alpha, N):
        self.t = t
        self.k = k
        self.s = s
        self.sigma = sigma
        self.r = r
        self.q = q
        self.eta = eta
        self.alpha = alpha
        self.N = N

    def gaussian_characteristic_func(self, v):
        mu = np.log(self.s) + (self.r - self.q - np.power(self.sigma, 2) / 2) * self.t
        vol = np.power(self.sigma, 2) * self.t
        return np.exp(1j * mu * v - vol * np.power(v, 2) / 2)

    def call_characteristic_func(self, w, coef, exp, v):
        return w*coef * exp * self.gaussian_characteristic_func(v)

    def fft_main(self):
        C = np.exp(-self.r * self.t)
        lam = 2 * np.pi / (self.N * self.eta)
        beta = np.log(self.s) - lam * self.N / 2
        vs = self.eta * np.arange(self.N)
        km = beta + np.arange(self.N) * lam

        coef = C / ((self.alpha + 1j * vs) * (self.alpha + 1j * vs + 1))
        exps = np.exp(-1j * beta * vs)

        ws = np.ones(self.N) * self.eta
        ws[0] = self.eta / 2
        # ws = SimpsonW(self.N, self.eta)

        fft_x = np.array(
            [self.call_characteristic_func(ws[i], coef[i], exps[i], vs[i] - (self.alpha + 1) * 1j) for i in range(len(vs))])

        y = np.fft.fft(fft_x)

        return np.exp(km), np.exp(-self.alpha * km) / np.pi * y.real


if __name__ == '__main__':
    params = {"s": 100.0,  # index level
              "k": 90.0,  # option strike
              "t": 1.0,  # maturity date
              "r": 0.05,  # risk-less short rate
              "sigma": 0.3,  # volatility
              "q": 0,
              "eta": 0.15,
              "alpha": 1,
              "N": np.power(2, 12)}
    eurofft = EuroOptionFFT(**params)

    km, p_fft = eurofft.fft_main()
    print(np.interp(90, km, p_fft))

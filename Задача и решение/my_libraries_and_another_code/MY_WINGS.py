#!/usr/bin/env python
# coding: utf-8

# Здесь хранятся изучаемые кралья

import numpy as np

class wing1:
    def __init__(self, ):
        """
        Инициализирует крыло первого типа
        """
        # эти данные получены кропотливым ручным програмно-экспериментальным трудом
        self.n_parts = 10
                
        self.zones_inds = []
        self.z1 = np.arange(0,833) # верхняя часть ближней передней кромки.    "ближняя" - ближе к корпусу самолёта
        self.z2 = np.arange(833, 13377) # нижняя часть первого блока крыла.    "первый" - отсчёт от корпуса самолёта
        self.z3 = np.arange(13377,15009) # нижняя часть дальней передней кромки
        self.z4 = np.arange(15009,27297) # нижняя часть второго блока крыла
        self.z5 = np.arange(27297,39585) # нижняя часть третьего блока крыла
        self.z6 = np.arange(39585,40369) # верхняя часть ближней передней кромки
        self.z7 = np.arange(40369,52913) # верхняя часть первого блока крыла
        self.z8 = np.arange(52913,54449) # верхняя часть дальней передней кромки
        self.z9 = np.arange(54449,66737) # верхняя часть второго блока крыла
        self.z10 = np.arange(66737,79025) # верхняя часть третьего блока крыла

        # возможно, перепутан верх и низ

        self.zones_inds.append(self.z1)
        self.zones_inds.append(self.z2)
        self.zones_inds.append(self.z3)
        self.zones_inds.append(self.z4)
        self.zones_inds.append(self.z5)
        self.zones_inds.append(self.z6)
        self.zones_inds.append(self.z7)
        self.zones_inds.append(self.z8)
        self.zones_inds.append(self.z9)
        self.zones_inds.append(self.z10)

        self.dims_zones = [(49, 17), (256, 49), (96, 17), (256, 48), (256, 48), 
                           (49, 16), (256, 49), (96, 16), (256, 48), (256, 48)]
        
        self.S_REF = 0.15


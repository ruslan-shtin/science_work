#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 14:03:34 2022

@author: student Shtin Ruslan
"""
import numpy as np
import sklearn.metrics

# для работы с моим интегратором
import my3D_surface_integrator as sint

def prediction_quality(predicted_cp, origin_cp, predicted_cx=None, predicted_cy=None, origin_cx=None, 
                       origin_cy=None, regime=None, XYZ_coords=None, wing=None, need_print=True, round_digit=10, debug=False, need_calculating_info=True):
    """
    Вычисление ошибок предсказания поля Cp и соответствующих значений Cx и Cy.


    Parameters
    ----------
    predicted_cp : DataFrame or Serias, shape=(Npoints, 1)
        Предсказанное поле Cp.
    origin_cp : DataFrame or Serias, shape=(Npoints, 1)
        Правильное поле Cp.
    predicted_cx : float, optional
        Предсказанное значение Cx. Если None, то вычисляется из predicted_cp интегрированием. 
        The default is None.
    predicted_cy : float, optional
        Предсказанное значение Cy. Если None, то вычисляется из predicted_cp интегрированием. 
        The default is None.
    origin_cx : float, optional
        Правильное значение Cx. Если None, то вычисляется из origin_cp интегрированием. 
        The default is None.
    origin_cy : float, optional
        Правильное значения Cy. Если None, то вычисляются из origin_cp интегрированием. 
        The default is None.
    regime : (float, float), optional
        Режим обтекания - число Маха набегающего потока и угол атаки. Нужен только для интегрирования.
        The default is None.
    XYZ_coords : DataFrame, shape=(Npoints, 3)
        Координаты точек крыла. Нужно только для интегрирования.
        The default is None.
    wing : wing1, optional
        Параметры крыла. Нужно только для интегрирования.
        The default is None.
    need_print : bool, optional
        Показывает, надо ли печатать значения ошибок. The default is True.
    round_digit : int, optional
        На сколько чисел округлять при выводе (нужен в случае need_print=True). The default is 10.
    debug : bool, optional
        Флаг, включающий вывод отладочной информации
    need_calculating_info : bool, optional
        Флаг, показывающий необходимость вывода информации о том, что сейчас считается

    Returns
    -------
    rmse_cp : float
        корень из средней квадратичной ошибки предсказания Cp
    mae_cp : float
        максимальная абсолютная ошибка предсказания Cp 
    ae_cx, ae_cy : float, float
        абсолютные ошибки предсказания Cx и Cy
    re_cx, re_cy : float, float
        относительные (respectived) ошибки предсказания Cx, Cy 
        (деление происходит на оригинальные значения)
    """
    predicted_cp = predicted_cp.to_numpy()
    predicted_cp = predicted_cp.reshape((-1,1))
    origin_cp = origin_cp.to_numpy()
    origin_cp = origin_cp.reshape((-1,1))
    # теперь это столбцы (Npoints, 1)
    
    rmse_cp = np.sqrt(sklearn.metrics.mean_squared_error(origin_cp, predicted_cp))
    mae_cp = sklearn.metrics.max_error(origin_cp, predicted_cp)
    
    if (predicted_cx is None) or (predicted_cy is None):
        assert regime is  not None, "Не указан Режим! Т.к. predicted_cx==None or predicted_cy==None, то надо считать интеграл."
        assert not(XYZ_coords is None), "Не указаны координаты крыла! Т.к. predicted_cx==None or predicted_cy==None, то надо считать интеграл."
        assert not(wing is None), "Не указано крыло! Т.к. predicted_cx==None or predicted_cy==None, то надо считать интеграл."
        
        if need_calculating_info:
            print("\tcalculating predicted_cx, predicted_cy")
            
        predicted_cx, predicted_cy, _ = sint.get_CxCyCz_SpdSys_byCp(xyz_coords=XYZ_coords, 
                                                                    Cp=predicted_cp, 
                                                                    Sref=wing.S_REF, 
                                                                    AngAt=regime[1], 
                                                                    zones_inds=wing.zones_inds, 
                                                                    dims_zones=wing.dims_zones, 
                                                                    indeces_for_inver=[], 
                                                                    Scalc=False)
        
    if (origin_cx is None) or (origin_cy is None):
        assert regime is not None, "Не указан Режим! Т.к. origin_cx==None or origin_cy==None, то надо считать интеграл."
        assert XYZ_coords is not None, "Не указаны координаты крыла! Т.к. origin_cx==None or origin_cy==None, то надо считать интеграл."
        assert wing is not None, "Не указано крыло! Т.к. origin_cx==None or origin_cy==None, то надо считать интеграл."
        
        if need_calculating_info:
            print("\tcalculating origin_cx, origin_cy")
        
        origin_cx, origin_cy, _ = sint.get_CxCyCz_SpdSys_byCp(xyz_coords=XYZ_coords, 
                                                              Cp=origin_cp, 
                                                              Sref=wing.S_REF, 
                                                              AngAt=regime[1], 
                                                              zones_inds=wing.zones_inds, 
                                                              dims_zones=wing.dims_zones, 
                                                              indeces_for_inver=[], 
                                                              Scalc=False)
    
    ae_cx = np.abs(predicted_cx - origin_cx)
    ae_cy = np.abs(predicted_cy - origin_cy)
    re_cx = ae_cx / np.abs(origin_cx)
    re_cy = ae_cy / np.abs(origin_cy)
    
    if need_print:
        print("RMSE Cp =", np.round(rmse_cp, round_digit))
        print("MAE  Cp =", np.round(mae_cp, round_digit))
        print("dCx = |Cx_true - Cx_pred| =", np.round(ae_cx, round_digit))
        print("dCy = |Cy_true - Cy_pred| =", np.round(ae_cy, round_digit))
        print("dCx / |Cx_true| =", np.round(re_cx, round_digit))
        print("dCy / |Cy_true| =", np.round(re_cy, round_digit))
    
    return (rmse_cp, mae_cp, ae_cx, ae_cy, re_cx, re_cy)


def seria_of_prediction_quality(set_predicted_cp, set_origin_cp, set_predicted_cx=None, set_predicted_cy=None, 
                                set_origin_cx=None, set_origin_cy=None, set_regime=None, XYZ_coords=None, wing=None,
                                need_print_every=False, need_print_total=True, round_digit=10, debug=False, progres_by_samples=True):
    """
    Вычисление средних ошибок предсказания поля Cp и соответствующих значений Cx и Cy.
    Надстройка для случая, когда имеется массив предсказаний

    Parameters
    ----------
    set_predicted_cp : DataFrame, shape=(Nsample, Npoints)
        Предсказанные поля Cp.
    set_origin_cp : DataFrame, shape=(Nsample, Npoints)
        Правильные поля Cp.
    set_predicted_cx : np.array(float), shape=(Nsample, ), optional
        Предсказанные значения Cx. Если None, то они вычисляются из predicted_cp интегрированием. 
        The default is None.
    set_predicted_cy : np.array(float), shape=(Nsample, ), optional
        Предсказанные значения Cy. Если None, то они вычисляются из predicted_cp интегрированием. 
        The default is None.
    set_origin_cx : np.array(float), shape=(Nsample, ), optional
        Правильные значения Cx. Если None, то они вычисляются из origin_cp интегрированием. 
        The default is None.
    set_origin_cy : np.array(float), shape=(Nsample, ), optional
        Правильные значения Cy. Если None, то они вычисляются из origin_cp интегрированием.
        The default is None.
    set_regime : np.array(M, alpha), shape=(Nsample, 2), optional
        Режимы обтекания - числа Маха набегающего потока и соответствующие углы атаки.
        Нужно только для интегрирования.
        The default is None.
    XYZ_coords : list(DataFrame), shape=(Nsample, Npoints, 3)
        Координаты точек крыла для каждого случая. Нужно только для интегрирования.
        The default is None.
    wing : wing1, optional
        Параметры крыла. Нужно только для интегрирования.
        The default is None.
    need_print_every : bool, optional
        Показывает, надо ли печатать значения ошибок на каждом тестовом примере.
        The default is False.
    need_print_total : bool, optional
        Показывает, надо ли печатать значения финальных осреднённых ошибок .
        The default is True.
    round_digit : int, optional
        На сколько чисел округлять при выводе (нужен в случае need_print=True).
        The default is 10.
    debug : bool, optional
        Флаг, включающий вывод отладочной информации. The default is False.
    progres_by_samples : bool, optional
        Флаг, показывающий, надо ли выводить информацию о прогрессе

    Returns
    -------
    rmse_cp : float
            среднее корней из средней квадратичной ошибки предсказания Cp
            
    mae_cp : float
            среднее максимальных абсолютных ошибок предсказания Cp 
        
    ae_cx, ae_cy : float, float
            среднее абсолютных ошибок предсказания Cx и Cy
        
    re_cx, re_cy : float, float
            среднее относительных (respectived) ошибок предсказания Cx, Cy 
            (деление происходит на оригинальные значения)  

    """
    Nsample = set_predicted_cp.shape[0]
    sets = [set_predicted_cx, set_predicted_cy, set_origin_cx, set_origin_cy, set_regime]
    
    for i, set_ in enumerate(sets):
        if set_ is None:
            sets[i] = np.array([None]*Nsample)
    # если параметр не был указан, то сделали соответствующий массив из None
    if debug:
        print("seria_of_prediction_quality:")
        print(f"\tset_predicted_cx = {sets[0]}")
        print(f"\tset_predicted_cy = {sets[1]}")
        print(f"\tset_origin_cx = {sets[2]}")
        print(f"\tset_origin_cy = {sets[3]}")
        print(f"\tset_regime = {sets[4]}")
        
    
    all_errors = np.zeros((Nsample, 6)) # 6 = количество возвращаемых параметров
    
    for i in range(Nsample):
        if progres_by_samples:
            print(f"seria_of_prediction_quality : sample {i}/{set_predicted_cp.shape[0]}")
        all_errors[i] = prediction_quality(predicted_cp=set_predicted_cp.iloc[i], 
                                           origin_cp=set_origin_cp.iloc[i], 
                                           predicted_cx=sets[0][i], 
                                           predicted_cy=sets[1][i], 
                                           origin_cx=sets[2][i],
                                           origin_cy=sets[3][i],
                                           regime=sets[4][i],
                                           XYZ_coords=XYZ_coords[i],
                                           wing=wing,
                                           need_print=need_print_every, 
                                           round_digit=round_digit,
                                           debug=debug,
                                           need_calculating_info=progres_by_samples)
    
    rmse_cp, mae_cp, ae_cx, ae_cy, re_cx, re_cy = np.mean(all_errors, axis=0)
    e_cx, e_cy = all_errors[:, 2], all_errors[:, 3]
    rmse_cx = np.sqrt(np.mean(e_cx**2))
    rmse_cy = np.sqrt(np.mean(e_cy**2))
    
    if need_print_total:
        print("mean RMSE Cp =", np.round(rmse_cp, round_digit))
        print("mean MAE  Cp =", np.round(mae_cp, round_digit))
        print("mean dCx = |Cx_true - Cx_pred| =", np.round(ae_cx, round_digit))
        print("mean dCy = |Cy_true - Cy_pred| =", np.round(ae_cy, round_digit))
        print("mean |dCx/Cx_true| =", np.round(re_cx, round_digit))
        print("mean |dCy/Cy_true| =", np.round(re_cy, round_digit))
        print("RMSE Cx =", np.round(rmse_cx, round_digit))
        print("RMSE Cy =", np.round(rmse_cy, round_digit))
    
    return (rmse_cp, mae_cp, ae_cx, ae_cy, re_cx, re_cy, rmse_cx, rmse_cy)
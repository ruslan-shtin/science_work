#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

# для построения суррогатных моделей
from sklearn.base import BaseEstimator

# LS - The Least square model
from smt.surrogate_models import LS, KRG

# RBF из SMT не подключается (из другой библиотеки - всё норм) [проблема именно на домашнем компьютере]
# from smt.surrogate_models import RBF

# для обрабтки входящих массивов
from sklearn.utils.validation import check_X_y, check_array

class DISTRIBUTION_PREDICTION(BaseEstimator):
    all_surrogate_models = {
        # все рассматриваемые суррогатные модели. Можно расширить этот список
        "LS" : LS,
        "KRG" : KRG,
        #"RBF" : RBF
    }
    
    def __init__(self, surrogate_model_name="LS", Npc=3, n_parts=1, parts = None, 
                 kw_args={'print_global':False}, hello_words=False):
        """
        модель, предсказывающая распределённый по крылу параметр
        поля мараметра каждого прецидента вытягиваются в 1D-вектор и затем комбинируются в мутрицу полей
        модель делает сингулярное разложение матрицы полей и выбирает Npc главных компонент. 
        Затем, суррогатные модели обучаются предсказывать коэффициенты, с которыми надо складывать
        выбранные главные компоненты, чтобы предсказать исходное распределение параметра по крылу 
        
        Input:
            surrogate_model_name - str - название используемой суррогатной модели. Пока доступны "LS", "KRG", "RBF"
            Npc - int - число используемых главных компонент
            n_parts - int - на сколько частей делить расчётную область. 
                            Для каждой части будет построена своя суррогатная модель.
                            Есть два крайних случая: 
                                n_parts=1, когда строится одна СМ над всем крылом
                                n_marts=число точек в расчётной сетке, тогда для каждой расчётной точки 
                                будет построена своя суррогатная модель
            parts - list(list(inds)) - Каждый элемент списка parts - это список, содержащий индексы, 
                                       относящиеся к какой-то части крыла.
                                       Если parts=`None`, то индексы распределяются равномерно, согласно n_parts
            kw_args -  словарь {'arg_name':arg_value} - словарь параметров для используемой суррогатной модели
            hello_words - bool - флаг. Надо ли выводить информацию при работе
        
        
        """
        assert parts == None or n_parts == len(parts), f"n_parts={n_parts} не совпадает с количеством частей, указанных в parts. Сделайте partas == None или n_parts == len(parts)"
        assert surrogate_model_name in self.all_surrogate_models.keys(), 'SM-name must be on of the following: {}'.format(self.all_surrogate_models.keys())
        
        if hello_words:
            print(f"{surrogate_model_name}, Npc={Npc} init")
        
        self.Npc = Npc    
        self.n_parts = n_parts
        self.parts = parts
        self.surrogate_model_name = surrogate_model_name
        self.kw_args = kw_args
        self.hello_words = hello_words
        
    def partition(self):
        """
        Создадим список списков индексов.
        По итогу работы функции вернётся список parts. 
        Каждый элемент списка parts - это список, содержащий индексы, относящиеся к какой-то части крыла.
        
        В случае, если n_parts=1, функция вернёт список всех точек крыла
        """
        n_elements_in_part = self.Nrows // self.n_parts
        parts = [np.arange(i*n_elements_in_part,(i+1)*n_elements_in_part) for i in range(self.n_parts-1)]
        parts.append(np.arange((self.n_parts-1)*n_elements_in_part,self.Nrows))
        return parts
        
    def get_basis_and_trainCoefficients_for_part(self, target_part):
        """
        Input:
            target_part - np.array (Nrows_part,Nsamples)
        Output:
            basis - np.array(Nrows_part,Npc)
            k - np.array(Nsamples,Npc)
        
        По итогу работы это функции вернутся матрица basis (Nrows_part,Npc), каждой столбец которой это главные 
        компнонеты, которые будут использоваться при предсказании, и матрица k (Nsamples, Npc), каждая строка 
        которой содержит коэффициенты разложения по главным компонентам.  
        """
        U, s, Vt = np.linalg.svd(target_part, full_matrices=False)
                
        # Npc базисных столбцов
        basis = U[:,:self.Npc]*s[:self.Npc]
        
        # теперь нужно получить Nsamples наборов из Npc коэффициентов для каждого из Nsamples режимов
        k = Vt[:self.Npc].T
        
        return basis, k
    
    def train_model_for_part(self, xtrain, ytrain):
        """
        Input:
            xtrain - np.array(Nsamples, 2) - вектор из пар (M, alpha)
            ytrain - np.array(Nsamples, Npc) - векотор целевых переменных-коэфициентов при главных компонентах
        Output:
            predictor - обученная суррогатная модель для части кр
        """        
        predictor = self.surrogate_model(**self.kw_args)
        predictor.set_training_values(xt=xtrain,yt=ytrain)
        predictor.train()
        
        return predictor
    
    def fit(self, regimes, target_T):
        """
        Обучение предиктора для предсказания по режиму потока распределения целевого параметра по крылу
        Input:
            regimes - DataFrame (Nsamples, 2) - столбец режимов. Каждая строчка содержит пару (M, alpha)
            target_T - DataFrame (Nsamples, Nrows) - каждая строка - это распределения целевого параметра 
                                                 по поверхности крыла при конкретном режиме
        Output:
            ничего. Просто обучает предиктор
        """
        regimes, target_T = check_X_y(regimes, target_T, multi_output=True)
        # regimes, target_T уже сконвертированы в np.array 
        
        target = target_T.T # target=DF(Nrows, Nsamples)
        
        self.surrogate_model = self.all_surrogate_models[self.surrogate_model_name]
        self.Predictors_n_parts = [] # список предикторов для каждой части крыла
                                     # каждый предиктор является суррогатной моделью
        self.basis_n_parts = [] # список базисов из главных компонент для каждой части крыла
        self.Nrows, self.Nsamples = target.shape
        
        if self.parts == None:
            self.parts = self.partition()
            
        
        for part in self.parts:
            target_part = target[part]
            
            basis, ytrain = self.get_basis_and_trainCoefficients_for_part(target_part)
            self.basis_n_parts.append(basis)
            
            predictor = self.train_model_for_part(xtrain=regimes, ytrain=ytrain)
            self.Predictors_n_parts.append(predictor)        
        return
    
    def predict_part(self, regimes, part_num):
        """
        Предсказание распределений целевой переменной на части крыла по известным режимам обтекания.
        Input:
            regimes - DataFrame (Npoints, 2) - столбец режимов. Каждая строчка содержит пару (M, alpha)
            part_num - int - номер части крыла, на которой надо сделать предсказание
        Output:
            target_part - DataFrame (Nrows_part,Npoints) - матрица распределений целевого значения
                                                           по части крыла. каждый столбец соответствует 
                                                           своему режиму.
        """        
        xtest = np.array(regimes)
        coeficients = (self.Predictors_n_parts[part_num]).predict_values(xtest)
        
        basis = self.basis_n_parts[part_num]
        prediction = basis @ coeficients.T
        
        target_part = pd.DataFrame(prediction, index=self.parts[part_num])
        return target_part
    
    def predict(self, regimes):
        """
        Предсказание распределения целевого параметра по всему крылу по известным режимам обтекания
        Input:
            regimes - DataFrame (Npoints, 2) - столбец режимов. Каждая строчка содержит пару (M, alpha)
        Output:
            target - DataFrame(Npoint, Nrows) - матрица распределений целевого параметра по всему крылу.
                                              Каждый столбец соответствует своему режиму.
        """
        regimes = check_array(regimes)
        list_of_target_parts = []
        for i in range(len(self.parts)):
            target_part = self.predict_part(regimes, i)
            list_of_target_parts.append(target_part)
            
        target = pd.concat(list_of_target_parts, axis=0)
        return target.T

class SOLVER_INTEGRALVALUES(BaseEstimator):
    all_surrogate_models = {
        "LS" : LS,
        "KRG" : KRG,
        #"RBF" : RBF
    }
    
    def __init__(self, surrogate_model_name="LS", kw_args={'print_global':False}, hello_words=False):
        """
        модель, предсказывающая 1, 2, или n параметров. 
        обыкновенная суррогатная модель
        Замечание: 
            для всех выходных значений будет использоваться одна и та же модель с одинаковыми гипер-параметрами. Модели будут просто по-разному 
            обучены.
        Input:
            surrogate_model_name -  str - название используемой суррогатной модели. Пока доступны "LS", "KRG", "RBF"
            kw_args -  словарь {'arg_name':arg_value} - словарь параметров для используемой суррогатной модели
            hello_words - bool - флаг. Надо ли выводить информацию при работеs
        
        """
        assert surrogate_model_name in self.all_surrogate_models.keys(), 'SM-name must be on of the following: {}'.format(self.all_surrogate_models.keys())

        if hello_words:
            print(f"{surrogate_model_name} init")
        
        self.surrogate_model_name = surrogate_model_name
        self.kw_args = kw_args
        self.hello_words = hello_words
        
    def fit(self, regimes, target):
        """
        Обучение предиктора на предсказание по режиму потока целевой интегральной величины
        Input:
            regimes - DataFrame (Nsamples, 2) - столбец режимов. Каждая строчка содержит пару (M, alpha)
            target - DataFrame (Nsamples, Nrows) - каждая строка - это значения целевых интегральных величин
        Output:
            ничего. Просто обучает предиктор
        """
        regimes, target = check_X_y(regimes, target, multi_output=True)
        # regimes, target уже сконвертированы в np.array 
        
        self.surrogate_model = self.all_surrogate_models[self.surrogate_model_name](**self.kw_args)
        
        self.surrogate_model.set_training_values(xt=regimes, yt=target)
        self.surrogate_model.train()   
        return
    
    def predict(self, regimes):
        """
        Предсказание целевых интегральных величин по известным режимам обтекания
        Input:
            regimes - DataFrame (Nsamples, 2) - столбец режимов. Каждая строчка содержит пару (M, alpha)
        Output:
            target - DataFrame(Nsamples, Nrows) - матрица целевых значений.
                                                  Каждый столбец соответствует своему режиму.
        """
        regimes = check_array(regimes)
    
        target = (self.surrogate_model).predict_values(regimes)
        return target



class CxCy_PREDICTOR(BaseEstimator):
    all_surrogate_models = {
        "LS" : LS,
        "KRG" : KRG,
        #"RBF" : RBF
    }
    
    def __init__(self, sm_names="LS", list_kw_args={'print_global':False}, coef_for_predict=['Cx', 'Cy'], debug=False):
        """
        Модель, предсказывающая силовые коэффициенты Cx и Cy. Позволляет использовать отдельную модель для Cx и отдельную модель для Cy.
        Например, можно использовать линейную регрессию (LS) для Cx и кригинг (KRG) для Cy.

        Parameters
        ----------
        sm_names : string or list of string, optional
            Названия используемых суррогатных моделей. Если len(sm_names)==1, то эта модель будет использована для предсказания всех коэффициентов. 
            The default is "LS".
        list_kw_args : словарь или список словарей, optional
            Гиперпараметры для используемых суррогатных моделей. Если len(list_kw_args)==1, Все модели будут использовать те же гиперпараметры. 
            The default is {'print_global':False}.
        coef_for_predict : list of strings
            Список названий предсказываемых параметров.
            The default is ['Cx', 'Cy'].
        debug : bool, optional
            Флаг, показывающий, надо ли выводить отладочную информацию. 
            The default is False.

        Returns
        -------
        None.
        """
        self.number_of_predict_params = len(coef_for_predict)
        self.coef_for_predict = coef_for_predict
        
        assert (type(sm_names) == list) or (type(sm_names) == str), 'Неподходящий тип параметра sm_names. Нужен string или list[string]'
        if type(sm_names) == list:
            for sm_name in sm_names:
                assert sm_name in self.all_surrogate_models.keys(), 'SM-name must be on of the following: {}'.format(self.all_surrogate_models.keys())
            
            assert self.number_of_predict_params % len(sm_names) == 0, 'Количество предсказываемых параметров и количество используемых суррогатных моделей не соотносятся (либо вы хотите использовать слишком много моделей, либо число предсказываемых коэффициентов не кратно числу моделей'
            
            expand_coef = self.number_of_predict_params // len(sm_names)
            sm_names *= expand_coef
        else :
            sm_names = [sm_names] * self.number_of_predict_params
            
        self.sm_names = sm_names
        
        
        assert (type(list_kw_args) == list) or (type(list_kw_args) == dict), 'Неподходящий тип параметра list_kw_args. Нужен dict или list[dict]'
        if type(list_kw_args) == list:
            assert self.number_of_predict_params % len(list_kw_args) == 0, 'Неверное количество параметров. Нужно, чтобы number_of_predict_params % len(list_kw_args) == 0'
            
            expand_coef = self.number_of_predict_params // len(list_kw_args)
            list_kw_args *= expand_coef
        else:
            list_kw_args = [list_kw_args] * self.number_of_predict_params
            
        self.list_kw_args = list_kw_args
        
        self.debug = debug
        
        if self.debug:
            print("CxCy_PREDICTOR initing...")
            print(f"\tsm_names = {sm_names}")
            print(f"\tlist_kw_args = {list_kw_args}")
            print(f"\tcoef_for_predict = {coef_for_predict}, number_of_predict_params = {self.number_of_predict_params}")

    def fit(self, regimes, target):
        """
        Обучение предиктора на предсказание по режиму потока целевых величин.

        Parameters
        ----------
        regimes : DataFrame or numpy.array, shape=(Nsamples, 2) 
            столбец режимов. Каждая строчка содержит пару (M, alpha).
        target : DataFrame or numpy.array, shape=(Nsamples, Nrows)
            каждая строка - это значения целевых величин.

        Returns
        -------
        None. Просто обучает модели.

        """
        regimes, target = check_X_y(regimes, target, multi_output=True)
        # regimes, target уже сконвертированы в np.array 
        
        if self.debug:
            print("CxCy_PREDICTOR fitting...")
            print(f"\tregimes.shape = {regimes.shape}")
            print(f"\ttarget.shape = {target.shape}")
        
        self.list_of_surrogate_models = []
        for i in range(self.number_of_predict_params):
            args = self.list_kw_args[i]
            sm_name = self.sm_names[i]
            tar_val = target[:,i]
            
            model = self.all_surrogate_models[sm_name](**args)
            model.set_training_values(xt=regimes, yt=tar_val)
            model.train()
            self.list_of_surrogate_models.append(model)
        return
    
    def predict(self, regimes):
        """
        Предсказание целевых величин по входным режимам обтекания

        Parameters
        ----------
        regimes : DataFrame or numpy.array. shape=(Nsamples, 2)
            столбец режимов. Каждая строчка содержит пару (M, alpha).

        Returns
        -------
        target : DataFrame (Nsamples, Nrows) 
            матрица целевых значений. Каждый столбец соответствует своему целевому значению
        """
        regimes = check_array(regimes)
        # теперь regimes сконвертирован в np.array
        
        if self.debug:
            print("CxCy_PREDICTOR predicting...")
            print(f"\tregimes.shape = {regimes.shape}")
        
        target = np.zeros((regimes.shape[0], self.number_of_predict_params))
        for i in range(self.number_of_predict_params):
            pred_col = self.list_of_surrogate_models[i].predict_values(regimes)
            target[:,i] = pred_col.ravel()
            
        target = pd.DataFrame(data=target, columns=self.coef_for_predict)
        return target

        

   

   

   

   

   

   

   





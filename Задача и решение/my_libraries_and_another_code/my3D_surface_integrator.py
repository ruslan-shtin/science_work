#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np

from sklearn.utils.validation import check_array, check_X_y

# # Общая часть2112

# ___Определение___ __интегрального поверхностного вектора__ :  
# Пусть у нас есть трёхмерная поверхность $S$, и скалярная величина $C_p$, распределённая по поверхности $S$, т.е. существует отображение $\mathbb{R}^3 \rightarrow \mathbb{R}$, которое каждой точке поверхности $S$ сопоставляет некоторое значение $C_p$.
# Для любой элементарной площадки $ds$ мы считаем, что величина $C_p$ на ней постоянна. Тогда каждой такой площадке $ds$ можно сопоставить вектор $d\vec{v}$, который сонаправлен с единичной внутренней нормалью к этой площадке, и модуль этого вектора равен $|d\vec{v}| = ds \cdot C_p$.
# Таким образом, если сложить все элементарные векторы $d\vec{v}$ (проинтегрировать по поверхности), то мы получим некоторый суммарный вектор $\vec{V}$, который назовём __интегральным поверхностным вектором__ . 
# 
# В силу векторной супер-позиции можно разбить исходную поверхность на конечное число непересекающихся поверхностей $\{S_1, S_2, \dots , S_n\}$, затем посчитать соответствующие интегральные поверхностные векторы $\vec{V}_1, \vec{V}_2, \dots, \vec{V}_n$, и тогда итоговый интегральный поверхностный вектор исходной поверхности $\vec{V}$ будет суммой полученных векторов : $\vec{V} = \sum\limits_{i=1}^n \vec{V}_i$

# In[20]:


def join_xyz_Cp(xyz_coords, Cp):
    """
    Обединяет в один 2D - массив
    
    Input:
        xyz_coords - np.array(Nrows, 3) - массив координат (каждая точка определяется тремя координатами)
        
        Cp - np.array(Nrows, 1) - массив локальных значений Cp
    Output:
        joined_data - np.array(Nrows, 4) - скомпанованный массив данных
    """
    xyz, cp = check_X_y(xyz_coords, Cp, multi_output=True)
    cp = np.reshape(cp, (xyz.shape[0], 1))
    
    joined_data = np.concatenate((xyz, cp),axis=1)
    return joined_data


# In[21]:


def get_integral_zone_normal(dims,data, inver_coef=1):
    """
    Считает интегральный поверхностный вектор для одной зоны, которая описана структурированной прямоугольной сеткой
    
    Или другими словами
    
    Считает поверхностный интеграл по параметически заданной обезразмеренной поверхности.
    Итогом интегрирования является вектор, являющийся суммой маленьких векторов v. 
    Вектор v имеет направление внутренней единичной нормали. Модуль вектора v равен произведению площади
    элементарной площадки на локальную величину коэффициента Cp (считаем, что на каждой маленькой площадке
    значение коэффициента постоянно)
    
    Input:
        dims - (int, int)=(number_of_period, point_in_one_period) 
             - кортеж размерности зоны (зона покрыта точками сторого периодически периодически 
               по прямоугольной сетке)
               
        data - np.array(x,y,z,Cp) 
             - массив данных. Для каждой точки зоны хранится её координата и значение коэффициента 
               давления Cp в ней
               
        inver_coef - int, 1 или -1 - коэффициент, отвечающий за инвертирование векторного произведения
    Output:
        s_ - float - площадь интегрируемой зоны
        
        sum_ - (float, float, float)=Cx, Cy, Cz - проекции интегрального вектора на оси лабораторной системы отсчёта.
    """
    i_max, j_max = dims
    sum_ = np.array([0.,0.,0.])
    s_ = 0
    
    for i in range(i_max-1):
        for j in range(j_max-1):
            c1=data[j   + i   *j_max]
            c2=data[j+1 + i   *j_max]
            c3=data[j   +(i+1)*j_max]
            c4=data[j+1 +(i+1)*j_max]
            aver = 0.25*(c1+c2+c3+c4) # в качестве рабочего значения коэф-та Cp в ячейке принимаем среднее значение
                                      # в угловых точках.
            
            S_vec_loc = inver_coef*0.5* np.cross((c4-c1)[:3],(c3-c2)[:3]) #площадь, четырёхугольника, 
                                                                          #выраженная через векторное произведение
            S_loc = np.dot(S_vec_loc,S_vec_loc)**0.5
            #при таком выражении каждая компонента S_vec_loc - это процекция площади на каждую из базисных плоскостей
            #поэтому, чтобы посчитать соответсвующие данной площадке Cx, Cy, Cz, надо просто домножить
            #Cp на каждую из компонент соответственно  
            
            s_ += S_loc
            sum_ += S_vec_loc*aver[-1]
    return s_, sum_

def get_integral_zone_CpCf(dims, data, inver_coef=1):
    """
    Считает интегральный поверхностный вектор(обусловленный и давлением и трением) для одной зоны, которая описана структурированной прямоугольной сеткой.


    Parameters
    ----------
    dims : (int, int)=(number_of_period, point_in_one_period)
        кортеж размерности зоны (зона покрыта точками сторого периодически периодически по прямоугольной сетке).
    data : np.array(x,y,z,Vx,Vy,Vx,Cp,Cf) [Nrows, 8]
        массив данных. Хранит информацию о каждой точке интегрируемой зоны.
    inver_coef : int, 1 or -1 , optional
        коэффициент, отвечающий за инвертирование векторного произведения. The default is 1.

    Returns
    -------
    s_ : float
        площадь интегрируемой зоны
    sum_ - np.array(float, float, float) = Cx, Cy, Cx
        проекции интегрального вектора на оси лабораторной системы отсчёта

    """
    i_max, j_max = dims
    sum_ = np.array([0.,0.,0.])
    s_ = 0
    
    for i in range(i_max-1):
        for j in range(j_max-1):
            c1=data[j   + i   *j_max]
            c2=data[j+1 + i   *j_max]
            c3=data[j   +(i+1)*j_max]
            c4=data[j+1 +(i+1)*j_max]
            aver = 0.25*(c1+c2+c3+c4) # в качестве рабочего значения коэф-та Cp и Cf и скорости граниченого потока в ячейке принимаем среднее значение
                                      # в угловых точках.
            
            S_vec_loc = inver_coef*0.5* np.cross((c4-c1)[:3],(c3-c2)[:3]) #вектор площади четырёхугольника, 
                                                                          #выраженный через векторное произведение
            
            #при таком выражении каждая компонента S_vec_loc - это процекция площади на каждую из базисных плоскостей
            #поэтому, чтобы посчитать соответсвующие данной площадке Cx, Cy, Cz, надо просто домножить
            #Cp на каждую из компонент соответственно  
            dCxCyCz_press = S_vec_loc*aver[-2]
            
            S_loc = np.dot(S_vec_loc,S_vec_loc)**0.5 #модуль площади четырёхугольника
            speed = aver[3:6]
            tau = speed  / np.sqrt(np.sum(speed**2))
            dCxCyCz_fric = tau * S_loc * aver[-1]
            
            sum_ += dCxCyCz_press + dCxCyCz_fric
            
            s_ += S_loc
            
    return s_, sum_


# In[22]:


def surface_normal_vector_integrator(xyz_coords, Cp, zones_inds, dims_zones, indeces_for_inver=[]):
    """
    Считает x,y,z-компоненты интегрального поверхностного вектора и площадь интегрируемой поверхности
    
    Считает x,y,z-компоненты интегрального вектора по распределению Cp по поверхности крыла. 
    Будут считаться интегралы по отдельным зонам, описанным в dims_zones, 
    а затем результаты этих интегрирований будут складываться.
    
    Input:
        xyz_coords - np.array(Nrows, 3) 
                   - массив координат, задающих поверхность (каждая точка определяется тремя координатами)
                   
        Cp - np.array(Nrows, 1) - массив локальных значений величины, распределённой по поверхности
        
        zones_inds - list(np.array(inds)) - список, содержащий индексы точек, для каждой зоны интегрирования
        
        dims_zones - list(pair(dim1, dim2)) - список размерностей зон интегрирования.
        
        indeces_for_inver - list(int) 
                          - список зон, в которых надо инверировать векторное произведение 
                            (может потребоваться при неподходящем обходе поверхности)
    Output:
        S - float - площадь интегрируемой поверхности
        (Cx, Cy, Cz) - (float,float,float) 
                     - x,y,z-компоненты интегрального поверхностного вектора (значения безразмерных коэффициентов)
    """
    
    # сначала данные нужно подготовить
    data = join_xyz_Cp(xyz_coords, Cp)
    # data - массив Nrows на 4. Первые три столбца - координаты. Четвёртый - Cp
    
    CxCyCz = np.array([0.,0.,0.])
    S = 0
    
    for i in range(len(zones_inds)):
        inds = zones_inds[i]
        dims = dims_zones[i]
        inver_coef = 1
        
        if i in indeces_for_inver:
            inver_coef = -1
        
        addS, addCxCyCz = get_integral_zone_normal(dims, data[inds], inver_coef)
        S += addS
        CxCyCz += addCxCyCz
    
    return S, CxCyCz

def surface_CpCf_integrator(xyz_coords, xyz_speeds, Cp, Cf, zones_inds, dims_zones, indeces_for_inver=[], progress_bar=True):
    """
    Считает x,y,z-компоненты интегрального поверностного вектора, равного сумме нормального и тангенсыального векторов

    Parameters
    ----------
    xyz_coords : np.array(Nrows, 3)
        массив координат, задающих поверхность (каждая точка определяется тремя координатами).
    xyz_speeds : np.array(Nrows, 3)
        массив скоростей над каждой из точек на поверхности (каждая скорость - это трёхмерный вектор).
    Cp : np.array(Nrows, 1)
        массив локальных значений Cp
    Cf : np.array(Nrows, 1)
        массив локальных значений Cf.
    zones_inds : list(np.array(inds))
        список, содержащий индексы точек, для каждой зоны интегрирования.
    dims_zones : list(pair(dim1, dim2))
        список, содержащий индексы точек, для каждой зоны интегрирования.
    indeces_for_inver : list(int), optional
        список зон, в которых надо инверировать векторное произведение (может потребоваться при неподходящем обходе поверхности). The default is [].
    progress_bar : bool, optional
        Флаг, говорящий, надо ли выводить информацию о прогрессе вычислений

    Returns
    -------
    S : float
        площадь интегрируемой поверхности
    (Cx, Cy, Cz) : np.array(float, float, float)
        x,y,z-компоненты интегрального поверхностного вектора (значения безразмерных коэффициентов)

    """
    # сначала данные нужно подготовить
    xyz_c = check_array(xyz_coords)
    xyz_s = check_array(xyz_speeds)
    cp = np.array(Cp)
    cp = np.reshape(cp, (-1, 1))
    cf = np.array(Cf)
    cf = np.reshape(cf, (-1, 1))

    data = np.concatenate((xyz_c, xyz_s, cp, cf),axis=1)
    
    # data - массив Nrows на 8. Первые три столбца - координаты. Седьмой - Cp ...
    
    CxCyCz = np.array([0.,0.,0.])
    S = 0.0
    
    for i in range(len(zones_inds)):
        if progress_bar:
            print(f"surface_CpCf_integrator: progress :  {i}/{len(zones_inds)-1}  \r", end="")
        inds = zones_inds[i]
        dims = dims_zones[i]
        inver_coef = 1
        
        if i in indeces_for_inver:
            inver_coef = -1
        
        addS, addCxCyCz = get_integral_zone_CpCf(dims, data[inds], inver_coef)
        S += addS
        CxCyCz += addCxCyCz
    if progress_bar:
        print()
    
    return S, CxCyCz

# # Специальная часть

# In[23]:
def get_CxCyCz_LabSys_byPresFric(xyz_coords, xyz_speeds, Cp, Cf, Sref, zones_inds, dims_zones, indeces_for_inver=[], Scalc=False, progress_bar=True):
    """
    Получение значений коэффициентов Cx, Cy, Cz в лабораторной системе отсчёта по распределению коэффициентов
    давления и трения по поверхности крыла.

    Parameters
    ----------
    xyz_coords : np.array(Nrows, 3)
        массив координат, задающих поверхность (каждая точка определяется тремя координатами).
    xyz_speeds : np.array(Nrows, 3)
        массив скоростей над каждой из точек на поверхности (каждая скорость - это трёхмерный вектор).
    Cp : np.array(Nrows, 1)
        массив локальных значений Cp
    Cf : np.array(Nrows, 1)
        массив локальных значений Cf.
    Sref : float
        номинальное значение площади крыла.
    zones_inds : list(np.array(inds))
        список, содержащий индексы точек, для каждой зоны интегрирования.
    dims_zones : list(pair(dim1, dim2))
        список, содержащий индексы точек, для каждой зоны интегрирования.
    indeces_for_inver : list(int), optional
        список зон, в которых надо инверировать векторное произведение (может потребоваться при неподходящем обходе поверхности). The default is [].
    Scalc : bool, optional
        флаг, показывающий надо ли считать площадь интегрируемой поверхности (она отличается от Sref примерно в два раза). The default is False.
    progress_bar : bool, optional
        Флаг, говорящий, надо ли выводить информацию о прогрессе вычислений

    Returns
    -------
    Cx, Cy, Cz : float,float,float
        x,y,z-компоненты интегрального поверхностного силового вектора, обусловленного давлением и трением (значения безразмерных коэффициентов)
    
    S : float
        площадь интегрируемой поверхности(должна отличаться от Sref примерно в два раза). Возвращается только в случае, если Scalc=True
    """
    s, CXCYCZ = surface_CpCf_integrator(xyz_coords=xyz_coords, xyz_speeds=xyz_speeds, Cp=Cp, Cf=Cf,
                                        zones_inds=zones_inds, dims_zones=dims_zones, indeces_for_inver=indeces_for_inver, progress_bar=progress_bar)
    cx, cy, cz = CXCYCZ/Sref
    
    if Scalc:
        return cx, cy, cz, s
    else:
        return cx, cy, cz
    


def get_CxCyCz_LabSys_byCp(xyz_coords, Cp, Sref, zones_inds, dims_zones, indeces_for_inver=[], Scalc=False):
    """
    Получение значений коэффициентов Cx, Cy, Cz в лабораторной системе отсчёта по распределению коэффициента
    давления по поверхности крыла.
    
    Input:
        xyz_coords - np.array(Nrows, 3) 
                   - массив координат, задающих поверхность (каждая точка определяется тремя координатами)
                   
        Cp - np.array(Nrows, 1) - массив локальных значений Cp
        
        Sref - float - номинальное значение площади крыла
        
        zones_inds - list(np.array(inds)) - список, содержащий индексы точек, для каждой зоны интегрирования
        
        dims_zones - list(pair(dim1, dim2)) - список размерностей зон интегрирования.
        
        indeces_for_inver - list(int) 
                          - список зон, в которых надо инверировать векторное произведение 
                            (может потребоваться при неподходящем обходе поверхности)
        
        Scalc - bool 
              - флаг, показывающий надо ли считать площадь интегрируемой поверхности 
                (она отличается от Sref примерно в два раза)
        
    Output:
        Cx, Cy, Cz - float,float,float 
                   - x,y,z-компоненты интегрального поверхностного вектора (значения безразмерных коэффициентов)
        
        S - float - площадь интегрируемой поверхности(должна отличаться от Sref примерно в два раза)
                    Возвращается только в случае, если Scalc=True
    """
    s, CXCYCZ = surface_normal_vector_integrator(xyz_coords=xyz_coords, Cp=Cp, zones_inds=zones_inds, 
                                                 dims_zones=dims_zones, indeces_for_inver=indeces_for_inver)
    cx, cy, cz = CXCYCZ/Sref
    
    if Scalc:
        return cx, cy, cz, s
    else:
        return cx, cy, cz


# __Лабораторная система отсчёта__:  Ось $X$ направлена вдоль профиля, ось $Y$ перпендикулярно ей вверх
# 
# __Скоростная система отсчёта__: Ось $X$ направлена по направлению набегающего потока, ось $Y$ перпендикулярно ей вверх (против направления сил тяжести)

# In[24]:


def LabSys2SpdSys_CxCyCz(Cx_lab, Cy_lab, Cz_lab, AngAt=0.0, debug=False):
    """
    Пересчитывает значения коэффициентов Cx, Cy, Cz в лабораторной системе отсчёта в скоростную систему отсчёта.
    
    Input:
        Cx_lab, Cy_lab, Cz_lab - float,float,float - значения коэффициентов Cx, Cy, Cz в 
                                                     лабораторной системе отсчёта
                                                     
        AngAt - float - угол атаки набегающего потовка (В ГРАДУСАХ)
        
        debug - bool - флаг показывающий режим отладки
    
    Output:
        Cx_spd, Cy_spd, Cz_spd - float,float,float - значения коэффициентов Cx, Cy, Cz в 
                                                     скоростной системе отсчёта
    """
    # здесь происходит прямой пересчёт, но это можно оформить в виде матричного перемножения
    
    if debug:
        print(f"LabSys2SpdSys_CxCyCz : start. \nInput:                 \nCx_lab={Cx_lab}, \nCy_lab={Cy_lab},\nCz_lab={Cz_lab},\nAngAt={AngAt}")
    
    
    
    SINA = np.sin(AngAt/180. * np.pi)
    COSA = np.cos(AngAt/180. * np.pi)
    
    if debug:
        print(f"\nSINA={SINA}, COSA={COSA}")
    
    
    Cx_spd = Cx_lab*COSA + Cy_lab*SINA
    Cy_spd =-Cx_lab*SINA + Cy_lab*COSA
    Cz_spd = Cz_lab
    
    if debug:
        print(f"LabSys2SpdSys_CxCyCz : finish. \n Output: Cx_spd={Cx_spd}, Cy_spd={Cy_spd}, Cz_spd={Cz_spd}")
    
    return Cx_spd, Cy_spd, Cz_spd


# In[25]:


def get_CxCyCz_SpdSys_byCp(xyz_coords, Cp, Sref, AngAt,zones_inds, dims_zones, indeces_for_inver=[], Scalc=False):
    """
    Получение значений коэффициентов Cx, Cy, Cz в скоростной системе отсчёта по распределению коэффициента
    давления по поверхности крыла.
    
    Input:
        xyz_coords - np.array(Nrows, 3) 
                   - массив координат, задающих поверхность (каждая точка определяется тремя координатами)
                   
        Cp - np.array(Nrows, 1) - массив локальных значений Cp
        
        Sref - float - номинальное значение площади крыла
        
        AngAt - float - угол атаки набегающего потовка (В ГРАДУСАХ)
        
        zones_inds - list(np.array(inds)) - список, содержащий индексы точек, для каждой зоны интегрирования
        
        dims_zones - list(pair(dim1, dim2)) - список размерностей зон интегрирования.
        
        indeces_for_inver - list(int) 
                          - список зон, в которых надо инверировать векторное произведение 
                            (может потребоваться при неподходящем обходе поверхности)
        
        Scalc - bool 
              - флаг, показывающий надо ли считать площадь интегрируемой поверхности 
                (она отличается от Sref примерно в два раза)
        
    Output:
        Cx, Cy, Cz - float,float,float 
                   - x,y,z-компоненты интегрального поверхностного вектора (значения безразмерных коэффициентов)
        
        S - float - площадь интегрируемой поверхности(должна отличаться от Sref примерно в два раза)
                    Возвращается только в случае, если Scalc=True
    """
    
    cxl, cyl, czl, s = get_CxCyCz_LabSys_byCp(xyz_coords=xyz_coords, Cp=Cp, Sref=Sref, 
                                              zones_inds=zones_inds, dims_zones=dims_zones, 
                                              indeces_for_inver=indeces_for_inver, Scalc=True)
    
    cx, cy, cz = LabSys2SpdSys_CxCyCz(Cx_lab=cxl, Cy_lab=cyl, Cz_lab=czl, AngAt=AngAt)
    
    if Scalc:
        return cx, cy, cz, s
    else:
        return cx, cy, cz


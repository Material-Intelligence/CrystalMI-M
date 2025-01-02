# -*- coding: utf-8 -*-
""" 
    This module defines the polygons.
    It requires modules: crystal_class, basis, CrystalLattice,
    ps_setd_mingled, mlab, tvtk and enclose_polyhedron.
"""
import ast

import numpy as np
import basis
from crystal_lattice import CrystalLattice
from cgeometry import enclosure
import crystal_class as CC
from crystal_shape import ps_setd_mingled
# from Inverse_Wulff_construction import crystal_plane, surface_areas, optimal_distance_local
import pandas as pd  # 添加导入

def crystal_plane(hkl, point_group_classnum, crystal_sys, param1, param2):
    '''
       return the geometric planes of the polyhedron, where the 32 point group is considered, 
       and the lattice parameter is taken as 0.4nm.
       hkl: Miller indices of crystallographic planes.
    '''


    class_name = f"CrystalClass{point_group_classnum}"
    b1 =get_b1(crystal_sys, param1, param2)

    # b1 = getattr(basis, crystal_sys)(a)
    cc1 = getattr(CC, class_name)()


    latt   = CrystalLattice(b1, cc1)
    planes = []
    number_planes = []
    for index in hkl:
        pf = latt.geometric_plane_family(index)
        number_planes.append(len(pf))
        planes.extend(pf)
    return planes, number_planes  # 确保只返回这两个值

def get_b1(crystal_sys, param1, param2):
    a, b, c = convert_string_to_numbers(param1)
    angle_a, angle_b, angle_c = convert_string_to_numbers(param2)
    class_sys = f"{crystal_sys}"
    if crystal_sys == "cubic":
        b1 = getattr(basis, class_sys)(a)
    elif crystal_sys == "orthorhombic":
        b1 = getattr(basis, class_sys)(a, b, c)
    elif crystal_sys == "hexagonal":
        b1 = getattr(basis, class_sys)(a, c)
    elif crystal_sys == "tetragonal":
        b1 = getattr(basis, class_sys)(a, c)
    elif crystal_sys == "monoclinic":
        b1 = getattr(basis, class_sys)(a, b, c, angle_b)
    elif crystal_sys == "trigonal":
        if angle_b == 56.75:
            b1 = getattr(basis, class_sys)(a, angle_a)
        else:
            class_sys = f"{crystal_sys}_hex"
            b1 = getattr(basis, class_sys)(a, c)
    elif crystal_sys == "triclinic":
        b1 = getattr(basis, class_sys)(a, b, c, angle_a, angle_b, angle_c)

    return b1

def surface_areas(d ):
    '''
    input:
        origin-to-planes distances.
		
	    volume_calc: the natural or unscaled volume of the polyhedron.
		area_p: areas of all the planes.
		points_p: interception points on the planes, grouped by plane.   
		
	return: 
		area_ps: the area of each plane family (sum of areas of surfaces of a plane family). 
    ''' 
    ps_setd_mingled(number_planes, planes, d)
    planes_c = [p.as_list() for p in planes]
    pr       = enclosure(planes_c)   
    volume_calc = pr[0]              # the natural or unscaled volume of the polyhedron.
    area_p      = pr[1]              # areas of all the planes.
    points_p    = pr[2]              # interception points on the planes, grouped by plane.                                        
    area_ps     = []                 # area of each plane family: sum of areas of surfaces of a plane family.
    index_start = 0
    for subset_n in number_planes:
        index_end   = index_start + subset_n
        area_ps.append(np.sum(area_p[index_start : index_end]))
        index_start = index_end  
    diameter = 10.0 
    volume   = 4.0/3.0*np.pi* (diameter*0.5)**3 
    area_ps  = np.array(area_ps)*np.power(volume/volume_calc, 2.0/3) 
	
    return area_ps

def euclidean_distance(surface_area1, surface_area2):
    # 计算欧氏距离
    return np.sqrt(np.sum((surface_area1 - surface_area2) ** 2))

def convert_str_to_list(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s


def call_basis(crystal, param):
    obj = basis()
    method = getattr(obj, crystal, None)
    if callable(method):
        method(param)

def call_CC(number):
    obj = CC()
    method_name = f"CrystalClass{number}"
    method = getattr(obj, method_name, None)
    if callable(method):
        method()

def convert_string_to_numbers(s: str):
    # 将字符串按空格分割为多个部分，并将每个部分转换为浮点数
    numbers = [float(part) for part in s.split()]

    # 使用解包将列表中的元素依次返回
    return tuple(numbers)

def predict_(df_eavl, df_point_information):
    # 这三行要根据预测的晶系、点群、晶面修改，和生成数据时的修改是一样的
    for i in df_eavl.index:
        crystal_sys = str(df_eavl['crystal'][i])
        point_group = str(df_eavl['pointgroup'][i])
        param = df_point_information['晶胞参数'][df_point_information['点群'] == point_group].iloc[0]
        a, b, c = convert_string_to_numbers(param)
        # b1 = call_basis(crystal_sys, a)
        # cc1 = call_CC(point_group)
        hkl = df_eavl['face'][
            i]  # [[2,-2, 0], [2,-2,2], [2,-2,4]]#[[2,-2,0], [Crystal_function, Crystal_function, -Crystal_function], [0, 2, 0]] #[[2,-2, 0], [0,2, 0]]

        print(hkl)
        # [[Crystal_function,Crystal_function,-Crystal_function], [0,2,0], [2,-2,4]]#[[2,-2,2], [2,-2,4]]

        # 这几行不要变
        diameter = 10.0
        volume = 4.0 / 3.0 * np.pi * (diameter * 0.5) ** 3
        planes, number_planes = crystal_plane(hkl)

        # 这个dist就是预测的表面能
        dist = df_eavl['dist_List'][i]  # energy [Crystal_function.000, Crystal_function.159,Crystal_function.111]#[Crystal_function.000,Crystal_function.028,Crystal_function.232] # [Crystal_function.000, 0.754]#[Crystal_function.000, Crystal_function.136]
        areas_predict = surface_areas(dist) / sum(surface_areas(dist))

        # areas_actual是你输入的表面积
        areas_actual = df_eavl['area_List'][i]  # [0.7533, 0.0071, 0.2394]#[0.1724, 0.8275]

        print('areas_predict', areas_predict)
        print('areas_actual', areas_actual)
        distance = euclidean_distance(areas_predict, areas_actual)

        print('distance', distance)
if __name__=="__main__" :
   # 读取CSV文件
    df_eavl = pd.read_csv('cubic_m3m_4.csv')
    df_eavl['face'] = df_eavl['face'].apply(convert_str_to_list)
    df_eavl['dist_List'] = df_eavl['dist_List'].apply(convert_str_to_list)
    df_eavl['area_List'] = df_eavl['area_List'].apply(convert_str_to_list)
    df_point_information = pd.read_excel("点群分类.xlsx", sheet_name='Sheet1')
    df_point_information['点群'].astype(str)
    # predict_(df, df_point_information)
    sum_distance =0
    for i in df_eavl.index:
        crystal_sys = str(df_eavl['crystal'][i])
        point_group = str(df_eavl['pointgroup'][i])
        param1 = df_point_information['晶胞参数'][df_point_information['点群'] == 'm-3m'].iloc[0]
        param2 = df_point_information['角度'][df_point_information['点群'] == 'm-3m'].iloc[0]
        point_group_classnum = df_point_information['class'][df_point_information['点群'] == 'm-3m'].iloc[0]
        # a, b, c = convert_string_to_numbers(param)
        # b1 = call_basis(crystal_sys, a)
        # cc1 = call_CC(point_group)
        hkl = df_eavl['face'][i]  # [[2,-2, 0], [2,-2,2], [2,-2,4]]#[[2,-2,0], [Crystal_function, Crystal_function, -Crystal_function], [0, 2, 0]] #[[2,-2, 0], [0,2, 0]]
        print(hkl)
        # [[Crystal_function,Crystal_function,-Crystal_function], [0,2,0], [2,-2,4]]#[[2,-2,2], [2,-2,4]]

        # 这几行不要变
        diameter = 10.0
        volume = 4.0 / 3.0 * np.pi * (diameter * 0.5) ** 3
        planes, number_planes = crystal_plane(hkl, point_group_classnum, crystal_sys, param1, param2)

        # 这个dist就是预测的表面能
        dist = df_eavl['dist_List'][i]  # energy [Crystal_function.000, Crystal_function.159,Crystal_function.111]#[Crystal_function.000,Crystal_function.028,Crystal_function.232] # [Crystal_function.000, 0.754]#[Crystal_function.000, Crystal_function.136]
        areas_predict = surface_areas(dist) / sum(surface_areas(dist))

        # areas_actual是你输入的表面积
        areas_actual = df_eavl['area_List'][i]  # [0.7533, 0.0071, 0.2394]#[0.1724, 0.8275]
        print('areas_predict', areas_predict)
        print('areas_actual', areas_actual)
        distance = euclidean_distance(areas_predict, areas_actual)
        sum_distance = sum_distance + distance
        print('distance', distance)

    avg_distance = sum_distance / len(df_eavl['area_List'])
    print('avg_distance', avg_distance)
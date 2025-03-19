import argparse
import os
import pandas as pd
import os
import csv
import json
from ast import literal_eval

def convert_face(s):
    result = []
    i = 0
    while i < len(s):
        if s[i] == '-':
            result.append(int(s[i] + s[i + 1]))
            i += 2
        else:
            result.append(int(s[i]))
            i += 1
    result = [result[i:i + 3] for i in range(0, len(result), 3)]
    return result


def clean_data(df, crystal_system, space_group, crystal_face):
    df_first_four_columns = df
    split = int(len(df_first_four_columns.columns) / 2)
    df_energy = df_first_four_columns.iloc[:, :-split].copy()
    df_area = df_first_four_columns.iloc[:, -split:].copy()
    df_energy['energy'] = df_energy.apply(lambda row: row.tolist(), axis=1)
    df_area['area'] = df_area.apply(lambda row: row.tolist(), axis=1)
    df1 = pd.DataFrame({
        'energy': df_energy['energy'],
        'area': df_area['area']
    })
    df1['crystal_system'] = crystal_system
    df1['point_group'] = space_group
    df1['crystal_face'] = [crystal_face] * len(df1)

    return df1

def load_data(directory):
    all_data = pd.DataFrame()
    filelist = os.listdir(directory)
    for i in range(len(filelist)):
        pointgroup = directory + '/' + filelist[i]
        for filename in os.listdir(pointgroup):
            if filename.endswith('.csv'):
                file_path = os.path.join(pointgroup, filename)
                data = pd.read_csv(file_path)
                file_name = file_path.split('\\')[-1].split('.')[0]
                parts = file_name.split('_')
                crystal_system = parts[0]  # 'cubic'
                space_group = parts[1]  # 'm-3'
                raw_face = parts[2]  # '2-202-22'
                crystal_face = convert_face(raw_face)
                df = clean_data(data, crystal_system, space_group, crystal_face)
                all_data = pd.concat([all_data, df], ignore_index=True)

    return all_data

def process_row(row, is_energy = False):
    crystal_system = row['crystal_system']
    point_group = row['point_group']
    crystal_face_str = str(row['crystal_face'])
    area_str = str(row['area'])
    energy_str = str(row['energy'])

    crystal_face_list = literal_eval(crystal_face_str)
    faces_str = ':'.join([str(face) for face in crystal_face_list])

    area_list = literal_eval(area_str)
    area_numbers = [int(a * 10000) for a in area_list]
    area_numbers_str = ':'.join([str(a) for a in area_numbers])

    energy_list = literal_eval(energy_str)
    energy_numbers = [int(e * 10000) for e in energy_list]
    energy_numbers_str = ':'.join([str(e) for e in energy_numbers])

    if is_energy:
        encoding = f"{crystal_system}({point_group})_face{{{faces_str}}}_area{{{area_numbers_str}}}"
        energy = f"energy{{{energy_numbers_str}}}"
        return {
            'encoding': encoding,
            'energy': energy
        }
    else:
        encoding = f"{crystal_system}({point_group})_face{{{faces_str}}}_energy{{{energy_numbers_str}}}"
        area = f"area{{{area_numbers_str}}}"
        return {
            'encoding': encoding,
            'area': area
        }

def process_csv_to_json(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        json_list = []
        for row in reader:
            json_obj = process_row(row)
            json_list.append(json_obj)

    with open(output_file, 'w', encoding='utf-8') as jsonfile:
        json.dump(json_list, jsonfile, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    directory = '../crystal_database/database/cubic_m-3m'
    df_cubic_m_3m = load_data(directory)
    random_columns = df_cubic_m_3m.sample(n=100)
    random_columns = random_columns.reset_index(drop=True)
    #build energy data
    json_list = []
    for i in range(len(random_columns)):
        row = random_columns.iloc[i]
        json_obj = process_row(row, is_energy = True)
        json_list.append(json_obj)
    output_file = 'preprocessed/energy_m-3m.json'
    with open(output_file, 'w', encoding='utf-8') as jsonfile:
        json.dump(json_list, jsonfile, ensure_ascii=False, indent=4)

    #build area data
    json_list = []
    for i in range(len(random_columns)):
        row = random_columns.iloc[i]
        json_obj = process_row(row, is_energy = False)
        json_list.append(json_obj)
    output_file = 'preprocessed/area_m-3m.json'
    with open(output_file, 'w', encoding='utf-8') as jsonfile:
        json.dump(json_list, jsonfile, ensure_ascii=False, indent=4)




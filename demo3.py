import os

def csv_to_list(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines if line != '']
    return txt_data


def read_csv(txt_path):
    txt_data = csv_to_list(txt_path)
    txt_data = [data.split(',') for data in txt_data]
    return txt_data

file = os.path.join('D:\\Datasets\\HCP_S1200', 'UKF_2T_AtlasSpace', 'tracts_commissural', '192439' + '.csv')
data = read_csv(file)
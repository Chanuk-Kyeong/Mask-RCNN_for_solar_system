import json
import os

dir = 'dataset/solar_generation_12cm_4387/meta'

file_list = os.listdir(dir)
for file in file_list:
    json_file_dir = dir+'/'+file
    with open(json_file_dir,'r',encoding='UTF8') as f:
        json_data = json.load(f)
        data_key = json_data['data_key']
        data_key = data_key[:-3]+"tif"
        json_data['data_key'] = data_key
    with open(json_file_dir, 'w', encoding='utf-8') as make_file:
        json.dump(json_data, make_file, indent="\t")
    old_filename = os.path.join(dir, file)
    new_filename = os.path.join(dir, file[:-8] +'tif.json')
    os.rename(old_filename,new_filename)
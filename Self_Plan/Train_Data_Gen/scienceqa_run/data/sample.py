import json

def extract_data(json_file, output_file, num_records):
    with open(json_file, 'r') as f:
        data = json.load(f)

    extracted_data = data[:num_records]

    with open(output_file, 'w') as f:
        json.dump(extracted_data, f)

# 指定JSON文件的路径和名称
json_file = '/data/rolnan/BOLAA/scienceqa_run/data/format_scienceqa.json'

# 指定输出文件的路径和名称
output_file = './sample_5.json'

# 指定要提取的记录数量
num_records = 5

# 调用函数进行数据提取和保存
extract_data(json_file, output_file, num_records)
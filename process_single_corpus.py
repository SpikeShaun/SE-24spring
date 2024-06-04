import pickle
from collections import Counter

# 加载pickle文件
def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='iso-8859-1')  # 使用iso-8859-1编码加载数据
    return data

# 将数据按是否为单一查询和多重查询分割
def split_data(total_data, qids):
    result = Counter(qids)  # 统计每个qid出现的次数
    total_data_single = []  # 存放单一查询的数据
    total_data_multiple = []  # 存放多重查询的数据
    for data in total_data:
        if result[data[0][0]] == 1:
            total_data_single.append(data)  # 如果qid出现次数为1，则是单一查询
        else:
            total_data_multiple.append(data)  # 否则是多重查询
    return total_data_single, total_data_multiple  # 返回分割后的数据

# 处理staqc数据
def data_staqc_processing(filepath, save_single_path, save_multiple_path):
    with open(filepath, 'r') as f:
        total_data = eval(f.read())  # 读取并解析文件中的数据
    qids = [data[0][0] for data in total_data]  # 提取所有数据的qid
    total_data_single, total_data_multiple = split_data(total_data, qids)  # 分割数据

    with open(save_single_path, "w") as f:
        f.write(str(total_data_single))  # 保存单一查询数据
    with open(save_multiple_path, "w") as f:
        f.write(str(total_data_multiple))  # 保存多重查询数据

# 处理大规模数据
def data_large_processing(filepath, save_single_path, save_multiple_path):
    total_data = load_pickle(filepath)  # 加载pickle文件
    qids = [data[0][0] for data in total_data]  # 提取所有数据的qid
    total_data_single, total_data_multiple = split_data(total_data, qids)  # 分割数据

    with open(save_single_path, 'wb') as f:
        pickle.dump(total_data_single, f)  # 保存单一查询数据到pickle文件
    with open(save_multiple_path, 'wb') as f:
        pickle.dump(total_data_multiple, f)  # 保存多重查询数据到pickle文件

# 将单一查询的未标注数据转化为标注数据
def single_unlabeled_to_labeled(input_path, output_path):
    total_data = load_pickle(input_path)  # 加载pickle文件
    labels = [[data[0], 1] for data in total_data]  # 为每个数据添加标签1
    total_data_sort = sorted(labels, key=lambda x: (x[0], x[1]))  # 按照qid和标签排序
    with open(output_path, "w") as f:
        f.write(str(total_data_sort))  # 保存标注后的数据

if __name__ == "__main__":
    # 配置staqc Python数据路径
    staqc_python_path = './ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    staqc_python_single_save = './ulabel_data/staqc/single/python_staqc_single.txt'
    staqc_python_multiple_save = './ulabel_data/staqc/multiple/python_staqc_multiple.txt'
    data_staqc_processing(staqc_python_path, staqc_python_single_save, staqc_python_multiple_save)  # 处理并保存staqc Python数据

    # 配置staqc SQL数据路径
    staqc_sql_path = './ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    staqc_sql_single_save = './ulabel_data/staqc/single/sql_staqc_single.txt'
    staqc_sql_multiple_save = './ulabel_data/staqc/multiple/sql_staqc_multiple.txt'
    data_staqc_processing(staqc_sql_path, staqc_sql_single_save, staqc_sql_multiple_save)  # 处理并保存staqc SQL数据

    # 配置大规模Python数据路径
    large_python_path = './ulabel_data/python_codedb_qid2index_blocks_unlabeled.pickle'
    large_python_single_save = './ulabel_data/large_corpus/single/python_large_single.pickle'
    large_python_multiple_save = './ulabel_data/large_corpus/multiple/python_large_multiple.pickle'
    data_large_processing(large_python_path, large_python_single_save, large_python_multiple_save)  # 处理并保存大规模Python数据

    # 配置大规模SQL数据路径
    large_sql_path = './ulabel_data/sql_codedb_qid2index_blocks_unlabeled.pickle'
    large_sql_single_save = './ulabel_data/large_corpus/single/sql_large_single.pickle'
    large_sql_multiple_save = './ulabel_data/large_corpus/multiple/sql_large_multiple.pickle'
    data_large_processing(large_sql_path, large_sql_single_save, large_sql_multiple_save)  # 处理并保存大规模SQL数据

    # 配置大规模SQL单一查询的标注数据保存路径
    large_sql_single_label_save = './ulabel_data/large_corpus/single/sql_large_single_label.txt'
    # 配置大规模Python单一查询的标注数据保存路径
    large_python_single_label_save = './ulabel_data/large_corpus/single/python_large_single_label.txt'
    single_unlabeled_to_labeled(large_sql_single_save, large_sql_single_label_save)  # 将SQL单一查询的未标注数据转化为标注数据
    single_unlabeled_to_labeled(large_python_single_save, large_python_single_label_save)  # 将Python单一查询的未标注数据转化为标注数据

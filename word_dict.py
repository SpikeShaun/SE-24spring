import pickle  # 导入pickle模块，用于序列化和反序列化数据

# 获取词汇表
def get_vocab(corpus1, corpus2):
    word_vocab = set()  # 创建一个空的集合，用于存储词汇
    for corpus in [corpus1, corpus2]:  # 遍历两个语料库
        for i in range(len(corpus)):  # 遍历语料库中的每一项
            word_vocab.update(corpus[i][1][0])  # 更新词汇集合，加入第一个子项的第一个元素
            word_vocab.update(corpus[i][1][1])  # 更新词汇集合，加入第一个子项的第二个元素
            word_vocab.update(corpus[i][2][0])  # 更新词汇集合，加入第二个子项的第一个元素
            word_vocab.update(corpus[i][3])  # 更新词汇集合，加入第三个子项
    print(len(word_vocab))  # 打印词汇集合的长度
    return word_vocab  # 返回词汇集合

# 加载pickle文件
def load_pickle(filename):
    with open(filename, 'rb') as f:  # 以二进制只读模式打开文件
        data = pickle.load(f)  # 反序列化数据
    return data  # 返回反序列化的数据

# 处理词汇表
def vocab_processing(filepath1, filepath2, save_path):
    with open(filepath1, 'r') as f:  # 以只读模式打开文件
        total_data1 = set(eval(f.read()))  # 读取并解析文件内容，将其转换为集合
    with open(filepath2, 'r') as f:  # 以只读模式打开文件
        total_data2 = eval(f.read())  # 读取并解析文件内容

    word_set = get_vocab(total_data2, total_data2)  # 获取词汇表

    excluded_words = total_data1.intersection(word_set)  # 找到两个集合的交集
    word_set = word_set - excluded_words  # 从词汇表中移除交集部分

    print(len(total_data1))  # 打印第一个词汇集合的长度
    print(len(word_set))  # 打印处理后的词汇表长度

    with open(save_path, 'w') as f:  # 以写模式打开文件
        f.write(str(word_set))  # 将处理后的词汇表写入文件

if __name__ == "__main__":
    # 配置文件路径
    python_hnn = './data/python_hnn_data_teacher.txt'
    python_staqc = './data/staqc/python_staqc_data.txt'
    python_word_dict = './data/word_dict/python_word_vocab_dict.txt'

    sql_hnn = './data/sql_hnn_data_teacher.txt'
    sql_staqc = './data/staqc/sql_staqc_data.txt'
    sql_word_dict = './data/word_dict/sql_word_vocab_dict.txt'

    new_sql_staqc = './ulabel_data/staqc/sql_staqc_unlabled_data.txt'
    new_sql_large = './ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.txt'
    large_word_dict_sql = './ulabel_data/sql_word_dict.txt'

    # 执行词汇表处理函数
    vocab_processing(sql_word_dict, new_sql_large, large_word_dict_sql)

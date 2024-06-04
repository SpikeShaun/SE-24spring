import pickle
import multiprocessing
from python_structured import *  # 导入解析Python代码和查询的模块
from sqlang_structured import *  # 导入解析SQL代码和查询的模块


# 多进程处理Python查询数据
def multipro_python_query(data_list):
    return [python_query_parse(line) for line in data_list]  # 解析每一行数据


# 多进程处理Python代码数据
def multipro_python_code(data_list):
    return [python_code_parse(line) for line in data_list]  # 解析每一行数据


# 多进程处理Python上下文数据
def multipro_python_context(data_list):
    result = []
    for line in data_list:
        if line == '-10000':
            result.append(['-10000'])  # 特殊标记处理
        else:
            result.append(python_context_parse(line))  # 解析上下文
    return result


# 多进程处理SQL查询数据
def multipro_sqlang_query(data_list):
    return [sqlang_query_parse(line) for line in data_list]  # 解析每一行数据


# 多进程处理SQL代码数据
def multipro_sqlang_code(data_list):
    return [sqlang_code_parse(line) for line in data_list]  # 解析每一行数据


# 多进程处理SQL上下文数据
def multipro_sqlang_context(data_list):
    result = []
    for line in data_list:
        if line == '-10000':
            result.append(['-10000'])  # 特殊标记处理
        else:
            result.append(sqlang_context_parse(line))  # 解析上下文
    return result


# 解析数据列表
def parse(data_list, split_num, context_func, query_func, code_func):
    pool = multiprocessing.Pool()  # 创建多进程池
    # 将数据列表分成若干子列表，每个子列表长度为split_num
    split_list = [data_list[i:i + split_num] for i in range(0, len(data_list), split_num)]

    # 多进程并行处理上下文数据
    results = pool.map(context_func, split_list)
    # 将子列表合并为一个完整列表
    context_data = [item for sublist in results for item in sublist]
    print(f'context条数：{len(context_data)}')  # 打印上下文数据条数

    # 多进程并行处理查询数据
    results = pool.map(query_func, split_list)
    # 将子列表合并为一个完整列表
    query_data = [item for sublist in results for item in sublist]
    print(f'query条数：{len(query_data)}')  # 打印查询数据条数

    # 多进程并行处理代码数据
    results = pool.map(code_func, split_list)
    # 将子列表合并为一个完整列表
    code_data = [item for sublist in results for item in sublist]
    print(f'code条数：{len(code_data)}')  # 打印代码数据条数

    pool.close()  # 关闭进程池
    pool.join()  # 等待所有进程完成

    return context_data, query_data, code_data  # 返回解析后的数据


# 主函数
def main(lang_type, split_num, source_path, save_path, context_func, query_func, code_func):
    # 加载源数据
    with open(source_path, 'rb') as f:
        corpus_lis = pickle.load(f)

    # 解析数据
    context_data, query_data, code_data = parse(corpus_lis, split_num, context_func, query_func, code_func)
    # 获取查询ID
    qids = [item[0] for item in corpus_lis]

    # 组合解析后的数据
    total_data = [[qids[i], context_data[i], code_data[i], query_data[i]] for i in range(len(qids))]

    # 保存解析后的数据
    with open(save_path, 'wb') as f:
        pickle.dump(total_data, f)


if __name__ == '__main__':
    # 配置Python数据路径
    staqc_python_path = '.ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    staqc_python_save = '../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.pkl'

    # 配置SQL数据路径
    staqc_sql_path = './ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    staqc_sql_save = './ulabel_data/staqc/sql_staqc_unlabled_data.pkl'

    # 处理并保存Python数据
    main(python_type, split_num, staqc_python_path, staqc_python_save, multipro_python_context, multipro_python_query,
         multipro_python_code)
    # 处理并保存SQL数据
    main(sqlang_type, split_num, staqc_sql_path, staqc_sql_save, multipro_sqlang_context, multipro_sqlang_query,
         multipro_sqlang_code)

    # 配置大规模Python数据路径
    large_python_path = './ulabel_data/large_corpus/multiple/python_large_multiple.pickle'
    large_python_save = '../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlable.pkl'

    # 配置大规模SQL数据路径
    large_sql_path = './ulabel_data/large_corpus/multiple/sql_large_multiple.pickle'
    large_sql_save = './ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.pkl'

    # 处理并保存大规模Python数据
    main(python_type, split_num, large_python_path, large_python_save, multipro_python_context, multipro_python_query,
         multipro_python_code)
    # 处理并保存大规模SQL数据
    main(sqlang_type, split_num, large_sql_path, large_sql_save, multipro_sqlang_context, multipro_sqlang_query,
         multipro_sqlang_code)

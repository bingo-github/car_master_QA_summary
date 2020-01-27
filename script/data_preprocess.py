# coding: UTF-8
'''
@author:    wujunbin342@163.com
@date:      2020-01-05
@desc:      data preprocess  数据预处理
'''
import sys
sys.path.append('script')

import re
import numpy as np
import pandas as pd
import jieba
from tqdm import tqdm
from multiprocessing import cpu_count, Pool


class DataPreprocess(object):
    def __init__(self, ori_data_fpath_list):
        jieba.load_userdict('./data/vocab/user_dict.txt')    # 添加自定义词典
        self.stopwords = set([line.strip() for line in open('./data/vocab/chinese_stopwords.txt', encoding='UTF-8').readlines()])   # 加载停用词
        self.ori_data_fpath_list = ori_data_fpath_list


    def _replace_special_char_(self, text):
        '''
        替换特殊字符
        :param text:  str 待替换的文本
        :return: str 替换后的文本
        '''
        text = text.lower()
        text = text.replace('.', '<PERIOD>')
        text = text.replace('。', '<PERIOD>')
        text = text.replace(',', '<COMMA>')
        text = text.replace('，', '<COMMA> ')
        text = text.replace('"', '<QUOTATION_MARK>')
        text = text.replace(';', '<SEMICOLON>')
        text = text.replace('!', '<EXCLAMATION_MARK>')
        text = text.replace('?', '<QUESTION_MARK>')
        text = text.replace('(', '<LEFT_PAREN>')
        text = text.replace(')', '<RIGHT_PAREN>')
        text = text.replace('--', '<HYPHENS>')
        text = text.replace('?', '<QUESTION_MARK>')
        text = text.replace(':', '<COLON>')
        return text


    def generate_vocab(self,
                       target_fpath='./data/vocab/vocab.txt',
                       text_columns=['Question', 'Dialogue', 'Report'],
                       special_text_columns=['Brand', 'Model'],
                       save_freq=True,
                       drop_lt_freq=None):
        '''
        根据原始数据生成词表

        :target_fpath: 词表保存地址
        :text_columns: 词表来源的字段，需要进行切词的字段
        :special_text_columns: 词表来源的字段，品牌或信号等不需要切词的字段
        :save_freq: bool 是否保存词频，由于词频后续可能有用，因此可以先保存
        :drop_lt_freq: int 去掉词频少于drop_lt_freq的词，如果为None，则不进行过滤
        :returns:

        >> data_preprocess_obj = DataPreprocess('./data/ori_data/AutoMaster_TrainSet.csv')
        >> data_preprocess_obj.generate_vocab(save_freq=False, drop_lt_freq=50)
        '''
        word_count_dict = {}

        # S1: 切词并获取数据
        for one_ori_data_fpath in self.ori_data_fpath_list:
            df = pd.read_csv(one_ori_data_fpath)
            target_text_columns_set = set(text_columns)&set(df.columns.to_list())
            target_special_text_columns_set = set(special_text_columns)&set(df.columns.to_list())
            for row in tqdm(df.itertuples(index=False), '[DataPreprocess.get_vocab] generating vocal from [{}]'.format(one_ori_data_fpath)):
                for one_col in target_text_columns_set:
                    text = getattr(row, one_col)
                    if pd.isna(text):
                        continue
                    text = self._replace_special_char_(text)
                    w_list = jieba.cut(text.replace('|', ' '))   # 由于在对话中，车主和技师的对话是用|分开的，为避免|的影响，将其替换为" "
                    for one_w in w_list:
                        word_count_dict.setdefault(one_w, 0)
                        word_count_dict[one_w] += 1
                for one_col in target_special_text_columns_set:   # 车品牌和型号不进行分词
                    spw = getattr(row, one_col)
                    if pd.isna(spw):
                        continue
                    word_count_dict.setdefault(spw, 0)
                    word_count_dict[spw] += 1

        # S2: 数据保存
        with open(target_fpath, 'w', encoding='utf-8') as fp:
            word_list = [word for word in word_count_dict.keys() if word not in self.stopwords and '\t' != word]
            for idx, one_w in enumerate(word_list):
                freq = word_count_dict[one_w]
                if drop_lt_freq is not None and freq < drop_lt_freq:     # 去掉低频词
                    continue
                one_vocab_info = [one_w, idx] if not save_freq else [one_w, idx, freq]
                fp.write('\t'.join(["{}".format(one_v) for one_v in one_vocab_info])+'\n')


    def load_dataset(self, train_data_fpath=None, test_data_fpath=None, verify_data_fpath=None):
        '''
        加载数据
        :param train_data_fpath:    str     训练集原始文件地址
        :param test_data_fpath:     str     测试集原始文件地址
        :param verify_data_fpath:   str     验证集原始文件地址
        :return:    train_data      DataFrame       训练集
                    test_data       DataFrame       测试集
                    verify_data     DataFrame       验证集
        '''
        train_data, test_data, verify_data = None, None, None
        # 分别读取训练，测试和验证集数据
        if train_data_fpath:
            train_data = pd.read_csv(train_data_fpath)
        if test_data_fpath:
            test_data = pd.read_csv(test_data_fpath)
        if verify_data_fpath:
            verify_data = pd.read_csv(verify_data_fpath)
        return train_data, test_data, verify_data


    def clean_sentence(self, sentence):
        '''
        去除特殊符号
        :param sentence:    str     待处理的字符串
        :return:    str     处理后的字符串
        '''
        if isinstance(sentence, str):
            return re.sub(
                r'[\s+\-\|\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】“”！，。？、~@#￥%……&*（）]+|车主说|技师说|语音|图片|你好|您好',
                '',
                sentence)
        else:
            return ""


    def drop_stopwords(self, words):
        '''
        去掉停用词
        :param words:   [word]
        :return:        [word]      去掉停用词后的词列表
        '''
        return [word for word in words if word not in self.stopwords]


    def sentence_proc(self, sentence):
        '''
        数据中对句子的处理方法
        :param sentence:    str     待处理的句子
        :return:    str     处理后的句子
        '''
        # 清除无用词
        sentence = self.clean_sentence(sentence)
        # 切词
        words = jieba.cut(sentence)
        # 过滤停用词
        words = self.drop_stopwords(words)
        # 拼接成一个字符串，以空格为分隔符
        return ' '.join(words)


    def data_frame_proc(self, df, cols=['Brand', 'Model', 'Question', 'Dialogue', 'Report']):
        '''
        数据批量处理方法
        :param df:      DataFrame   待处理的数据
        :param cols:    [str]       待处理的字段
        :return:        DataFrame   处理好的数据
        '''
        df = df.fillna('')
        for one_col in cols:
            if one_col in df.columns:
                df[one_col] = df[one_col].apply(self.sentence_proc)
        return df


    def parallelize(self, df, func):
        '''
        多核并行处理模块
        :param df:  DataFrame   待处理的数据
        :param func:    function    预处理函数
        :return:    DataFrame   处理后的数据
        '''
        if not hasattr(self, 'cores'):
            self.cores = cpu_count()
        if not hasattr(self, 'partitions'):
            self.partitions = self.cores
        # 数据切分
        data_split = np.array_split(df, self.partitions)
        # 进程池
        pool = Pool(self.cores)
        # 数据分发 合并
        data = pd.concat(pool.map(func, data_split))
        # 关闭进程池
        pool.close()
        # 执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
        pool.join()
        return data


    def get_wv_train_data(self, ori_data_fpath_list,
                                seg_data_fpath_list,
                                wv_train_data_fpath,
                                multi_process=False):
        '''
        获取词向量训练数据
        :param ori_data_fpath_list:  [str]  原始数据文件地址
        :param seg_data_fpath_list:  [str]  分词处理后的数据存储地址
        :param wv_train_data_fpath:  str    词向量训练数据存储地址
        :param multi_process:       bool, default False, 是否使用多进程
        :return:    DataFrame
        '''
        # 加载数据
        train_df, test_df, _ = self.load_dataset(ori_data_fpath_list[0],
                                                ori_data_fpath_list[1])
        if multi_process:
            train_df = self.parallelize(train_df, self.data_frame_proc)
            test_df = self.parallelize(test_df, self.data_frame_proc)
        else:
            train_df = data_preprocess_obj.data_frame_proc(train_df)
            test_df = data_preprocess_obj.data_frame_proc(test_df)
        train_df.to_csv(seg_data_fpath_list[0], index=None, header=True)
        test_df.to_csv(seg_data_fpath_list[1], index=None, header=True)

        train_df['merged'] = train_df[['Question', 'Dialogue', 'Report']].apply(lambda x: ' '.join(x), axis=1)
        test_df['merged'] = test_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
        merged_df = pd.concat([train_df[['merged']], test_df[['merged']]], axis=0)
        merged_df.to_csv(wv_train_data_fpath, index=None, header=False)
        return merged_df


if __name__ == "__main__":
    '''
    # 第一周作业：生成词表
    ori_data_fpath_list = ['./data/ori_data/AutoMaster_TrainSet.csv', './data/ori_data/AutoMaster_TestSet.csv']
    data_preprocess_obj = DataPreprocess(ori_data_fpath_list=ori_data_fpath_list)
    data_preprocess_obj.generate_vocab(save_freq=False, drop_lt_freq=5)
    '''

    # 第二周作业：生成训练词向量用的训练数据
    ori_data_fpath_list = ['./data/ori_data/AutoMaster_TrainSet.csv',
                           './data/ori_data/AutoMaster_TestSet.csv']
    seg_data_fpath_list = ['./data/processed_data/train_seg_data.csv',
                           './data/processed_data/test_seg_data.csv']
    wv_train_data_fpath = './data/processed_data/merge_train_test_seg_data.csv'
    data_preprocess_obj = DataPreprocess(ori_data_fpath_list=ori_data_fpath_list)
    merge_df = data_preprocess_obj.get_wv_train_data(ori_data_fpath_list,
                                          seg_data_fpath_list,
                                          wv_train_data_fpath,
                                          multi_process=False)


# coding: UTF-8
'''
@author:    wujunbin342@163.com
@date:      2020-01-05
@desc:      data preprocess  数据预处理
'''
import os
import sys
sys.path.append('script')

import pandas as pd
import jieba
from tqdm import tqdm


class DataPreprocess(object):
    def __init__(self, ori_data_fpath):
        jieba.load_userdict('./data/vocab/user_dict.txt')
        self.stopwords = set([line.strip() for line in open('./data/vocab/chinese_stopwords.txt', encoding='UTF-8').readlines()])
        self.ori_data_fpath = ori_data_fpath


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
                       target_fpath='./data/vocab/vacab.txt',
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

        >>> os.chdir('..')
        >>> data_preprocess_obj = DataPreprocess('./data/ori_data/AutoMaster_TrainSet.csv')
        >>> data_preprocess_obj.generate_vocab(save_freq=False, drop_lt_freq=50)
        '''
        word_count_dict = {}

        # S1: 切词并获取数据
        df = pd.read_csv(self.ori_data_fpath)
        for row in tqdm(df.itertuples(index=False), '[DataPreprocess.get_vocab] generating vocal'):
            for one_col in text_columns:
                text = getattr(row, one_col)
                if pd.isna(text):
                    continue
                text = self._replace_special_char_(text)
                w_list = jieba.cut(text.replace('|', ' '))   # 由于在对话中，车主和技师的对话是用|分开的，为避免|的影响，将其替换为" "
                for one_w in w_list:
                    word_count_dict.setdefault(one_w, 0)
                    word_count_dict[one_w] += 1
            for one_col in special_text_columns:   # 车品牌和型号不进行分词
                spw = getattr(row, one_col)
                if pd.isna(spw):
                    continue
                word_count_dict.setdefault(spw, 0)
                word_count_dict[spw] += 1

        # S2: 数据保存
        with open(target_fpath, 'w', encoding='utf-8') as fp:
            word_list = [word for word in word_count_dict.keys() if word not in self.stopwords]
            for idx, one_w in enumerate(word_list):
                freq = word_count_dict[one_w]
                if drop_lt_freq is not None and freq < drop_lt_freq:     # 去掉低频词
                    continue
                one_vocab_info = [one_w, idx] if not save_freq else [one_w, idx, freq]
                fp.write(' '.join(["{}".format(one_v) for one_v in one_vocab_info])+'\n')


if __name__ == "__main__":
    os.chdir('..')
    data_preprocess_obj = DataPreprocess('./data/ori_data/AutoMaster_TrainSet.csv')
    data_preprocess_obj.generate_vocab(save_freq=False, drop_lt_freq=50)

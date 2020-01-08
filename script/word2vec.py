# coding: UTF-8
'''
@author:    wujunbin342@163.com
@date:      2020-01-07
@desc:      word2vec的CBOW和skip-gram实现
'''
import sys
sys.path.append('./script')

import numpy as np
import pandas as pd
import jieba
from tqdm import tqdm
from collections import Counter


class Word2Vec(object):
    '''
    word2vec词向量模型
    '''
    def __init__(self,
                 corpus_fpath,
                 corpus_text_columns,
        vocab_fpath='./data/vocab/vocab.txt'):
        self.stopwords = set([line.strip() for line in open('./data/vocab/chinese_stopwords.txt', encoding='UTF-8').readlines()])
        self.corpus_fpath = corpus_fpath
        self.corpus_text_columns = corpus_text_columns
        self.vocab_fpath = vocab_fpath
        self.word_to_int = {}
        self.int_to_word = {}
        self.corpus_by_int = None


    def get_vocab(self):
        '''
        获取词表信息
        :return:
        '''
        with open(self.vocab_fpath, 'r') as fp:
            for line in fp:
                items = line.strip('\n').strip('\r').split(' ')
                self.word_to_int[items[0]] = int(items[1])
                self.int_to_word[int(items[1])] = items[0]


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


    def _get_corpus_words_(self):
        '''
        获取语料中的文本（对话文本）,并进行去停用词处理, 切词返回
        :return: words_list [word]
        '''
        words_list = []
        df = pd.read_csv(self.corpus_fpath)
        for row in tqdm(df.itertuples(index=False), '[DataPreprocess.get_vocab] generating vocal'):
            for one_col in self.corpus_text_columns:
                text = getattr(row, one_col)
                if pd.isna(text):
                    continue
                text = self._replace_special_char_(text)
                w_list = jieba.cut(text.replace('|', ' '))   # 由于在对话中，车主和技师的对话是用|分开的，为避免|的影响，将其替换为" "
                words_list.extend(w_list)
        return words_list


    def _drop_high_freq_words_(self, wordint_list, th, t=0.001):
        '''
        去掉高频词
        :param wordint_list:  [int]
        :param th:  float
        :param t: float
        :return:  [int]
        '''
        # 统计单词出现频次
        wordint_freq_dict = Counter(wordint_list)
        wordint_num = len(wordint_list)
        # 计算单词频率
        word_freqratio = {w: c / wordint_num for w, c in wordint_freq_dict.items()}
        # 计算被删除的概率
        prob_drop = {w: 1 - np.sqrt(t / word_freqratio[w]) for w in wordint_freq_dict}
        # 对单词进行采样
        wordint_list_after_dop = [w for w in wordint_list if prob_drop[w] < th]
        return wordint_list_after_dop


    def get_corpus_by_int(self, th_for_drop_high_freq_words=None):
        '''
        获取训练语料，并根据词表转化为int
        :th_for_drop_high_freq_words: float 去掉高频词的阈值。如果为float，则按照th进行过滤，如果为None，则不过滤
        :return:  wordint_list [int]
        '''
        self.get_vocab()
        words_list = self._get_corpus_words_()   # 获取语料中的文本（对话文本）,并进行去停用词处理
        wordint_list = [self.word_to_int[word] for word in words_list if word not in self.stopwords and word in self.word_to_int]
        if th_for_drop_high_freq_words:
            wordint_list = self._drop_high_freq_words_(wordint_list, th_for_drop_high_freq_words)
        return wordint_list


    def _get_target_(self, words, idx, window_size=5):
        '''
        获得input word的上下文单词列表
        :param words: [int] 输入的语料词
        :param idx: int 当前所选词的下标
        :param window_size: int 窗口
        :return: [int]
        '''
        target_window = np.random.randint(1, window_size+1)
        # 考虑当前词的前面单词不够的情况
        start_point = idx-target_window if (idx-target_window) > 0 else 0
        end_point = idx+target_window
        # 窗口中的上下文单词
        targets = set(words[start_point:idx]+words[idx+1:end_point+1])
        return targets


    def get_batch(self, words, batch_size, window_size=5):
        '''
        构造一个获取batch的生成器
        :param words: [int]
        :param batch_size: int batch大小
        :param window_size:  int 窗口大小
        :return:  (int, int)  (网络输入，网络输出)
        '''
        n_batches = len(words) // batch_size
        # 仅取full batches
        words = words[:n_batches*batch_size]
        for idx in range(0, len(words), batch_size):
            x, y = [], []
            batch = words[idx:idx+batch_size]
            for i in range(len(batch)):
                batch_x = batch[i]
                batch_y = self._get_target_(batch, i, window_size)
                # 由于一个input word会对应多个output word，因此需要长度统一
                x.extend([batch_x]*len(batch_y))
                y.extend((batch_y))
            yield x, y
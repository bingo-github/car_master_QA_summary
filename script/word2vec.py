# coding: UTF-8
'''
@author:    wujunbin342@163.com
@date:      2020-01-07
@desc:      word2vec的CBOW和skip-gram实现
            使用tensorflow 1.0
'''
import sys
sys.path.append('./script')
import time

import numpy as np
import pandas as pd
import jieba
from tqdm import tqdm
from collections import Counter
import tensorflow as tf


class Word2Vec(object):
    '''
    word2vec词向量模型
    '''
    def __init__(self,
                 corpus_fpath_list,
                 corpus_text_columns,
                 vocab_fpath='./data/vocab/vocab.txt',
                 embedding_size=300):
        self.stopwords = set([line.strip() for line in open('./data/vocab/chinese_stopwords.txt', encoding='UTF-8').readlines()])
        self.corpus_fpath_list = corpus_fpath_list
        self.corpus_text_columns = corpus_text_columns
        self.vocab_fpath = vocab_fpath
        self.embedding_size = embedding_size
        self.word_to_int = {}
        self.int_to_word = {}
        self.corpus_by_int = None


    def get_vocab(self):
        '''
        获取词表信息
        :return:
        '''
        with open(self.vocab_fpath, 'r', encoding='utf-8') as fp:
            for line in fp:
                items = line.strip('\n').strip('\r').split('\t')
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
        for one_corpus_fpath in self.corpus_fpath_list:
            df = pd.read_csv(one_corpus_fpath)
            target_corpus_text_columns_set = set(self.corpus_text_columns)&set(df.columns.to_list())
            for row in tqdm(df.itertuples(index=False), '[DataPreprocess.get_vocab] getint corpus from [{}]'.format(one_corpus_fpath)):
                for one_col in target_corpus_text_columns_set:
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


    def train_skip_gram(self, batch_size=100, window_size=3, epochs=10, n_sampled=1000):
        '''
        训练 skip-gram
        :param batch_size: int batch大小
        :param window_size: int 窗口大小
        :param epochs: int 迭代轮数
        :param n_sampled: int 采样数
        :return:

        >> word_vec_obj = Word2Vec(corpus_fpath_list=['./data/ori_data/AutoMaster_TrainSet.csv',
                                               './data/ori_data/AutoMaster_TestSet.csv'],
                            corpus_text_columns=['Question', 'Dialogue', 'Report'])
        >> word_vec_obj.train_skip_gram()
        '''
        # S1: 获取训练数据
        self.get_vocab()   # 获取此表数据
        wordint_list = self.get_corpus_by_int(th_for_drop_high_freq_words=0.8)

        # S2: 构建训练网络
        self.skip_gram_graph = tf.Graph()
        with self.skip_gram_graph.as_default():
            # 输入层：配置输入占位符
            inputs = tf.placeholder(tf.int32, shape=[None], name='inputs')
            labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')
            # 嵌入层
            self.vocab_size = len(self.word_to_int)
            embedding_w = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1, 1))
            embed = tf.nn.embedding_lookup(embedding_w, inputs)
            # 输出下采样
            softmax_w = tf.Variable(tf.truncated_normal([self.vocab_size, self.embedding_size], stddev=0.1))
            softmax_b = tf.Variable(tf.zeros(self.vocab_size))
            loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b, labels, embed, n_sampled, self.vocab_size)
            cost = tf.reduce_mean(loss)
            optimizer = tf.train.AdamOptimizer().minimize(cost)

        # S3: 训练
        with self.skip_gram_graph.as_default():
            saver = tf.train.Saver()
        with tf.Session(graph=self.skip_gram_graph) as sess:
            iteration = 1
            total_loss = 0
            sess.run(tf.global_variables_initializer())
            for e in range(1, epochs+1):
                # 获取batch数据
                batches = self.get_batch(words=wordint_list, batch_size=batch_size, window_size=window_size)
                start = time.time()
                for x, y in batches:
                    feed = {inputs:x, labels:np.array(y)[:, None]}
                    train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)
                    total_loss += train_loss
                    if 0 == iteration%100:
                        end = time.time()
                        print("Epoch {}/{}".format(e, epochs),
                              "Iteration: {}".format(iteration),
                              "Avg. Training loss: {:.4f}".format(total_loss / 1000),
                              "{:.4f} sec/batch".format((end - start) / 1000))
                        total_loss = 0
                        start = time.time()
                    iteration += 1

        # S4: 模型保存
        save_path = saver.save(sess, "./model/word2vec/skip_gram/checkpoints/text8.ckpt")


if __name__ == '__main__':
    word_vec_obj = Word2Vec(corpus_fpath_list=['./data/ori_data/AutoMaster_TrainSet.csv',
                                               './data/ori_data/AutoMaster_TestSet.csv'],
                            corpus_text_columns=['Question', 'Dialogue', 'Report'])
    word_vec_obj.train_skip_gram()

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
import json
import jieba
from tqdm import tqdm
from collections import Counter
import tensorflow as tf
from gensim.models import fasttext, word2vec
from gensim.models.word2vec import LineSentence

import data_preprocess


class MyWord2Vec(object):
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

        >> my_word_vec_obj = MyWord2Vec(corpus_fpath_list=['./data/ori_data/AutoMaster_TrainSet.csv',
                                               './data/ori_data/AutoMaster_TestSet.csv'],
                            corpus_text_columns=['Question', 'Dialogue', 'Report'])
        >> my_word_vec_obj.train_skip_gram()
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


class MyWord2Vev2(object):
    '''
    词向量化相关
    '''
    def __init__(self, wv_config_fpath,
                        wv_type='word2vec'):
        with open(wv_config_fpath, 'r') as fp:
            self.wv_config = json.load(fp)
        self.wv_type = wv_type


    def train_wv(self, merge_seg_data_fpath):
        '''
        训练词向量
        :param merge_seg_data_fpath: str 训练词向量的数据
        :return:
        '''
        # 训练词向量
        if 'word2vec' == self.wv_type:
            self.wv_model = word2vec.Word2Vec(LineSentence(merge_seg_data_fpath),
                                              min_count=self.wv_config['min_count'],
                                              size=self.wv_config['size'])
        elif 'fasttext' == self.wv_type:
            self.wv_model = fasttext.FastText(LineSentence(merge_seg_data_fpath),
                                              min_count=self.wv_config['min_count'],
                                              size=self.wv_config['size'])


    def load_vocab(self, vocab_fpath):
        '''
        加载词表
        :param vocab_fpath: str 文件路径
        :return: vocab_idx_map, idx_vocab_map
        '''
        with open(vocab_fpath, 'r') as fp:
            for line in fp:
                items = line.strip('\n').strip('\r').split('\t')
                word = items[0]
                idx = int(items[1])
                vocab_idx_map[word] = idx
                idx_vocab_map[idx] = word
        return vocab_idx_map, idx_vocab_map


    def expand_vocab(self, vocab_fpath, 
                    ori_train_fpath, 
                    ori_test_fpath, 
                    pad_trainX_fpath, 
                    pad_trainY_fpath, 
                    pad_testX_fpath):
        '''
        扩展wv模型的此表
        :param vocab_fpath: str 词表的文件地址
        :param ori_train_fpath:  str 文件地址
        :param ori_test_fpath:  str 文件地址
        :param pad_trainX_fpath:  str 文件地址
        :param pad_trainY_fpath:  str 文件地址
        :param pad_testX_fpath:  str 文件地址
        '''
        # 加载原先已经分词好的数据
        train_df = pd.read_csv(ori_train_fpath)
        test_df = pd.read_csv(ori_test_fpath)
        trainX_series = train_df[['Question', 'Dialogue']].apply(lambda x:' '.join(x), axis=1)
        trainY_series = train_df[['Report']]
        testX_series = test_df[['Question', 'Dialogue']].apply(lambda x:' '.join(x), axis=1)
        # 获取max_len
        trainX_max_len = data_preprocess.get_max_len(trainX_series)
        y_max_len = data_preprocess.get_max_len(trainY_series)
        testX_max_len = data_preprocess.get_max_len(testX_series)
        x_max_len = max(trainX_max_len, testX_max_len)
        # 获取词表
        vocab_idx_map, _ = self.load_vocab(vocab_fpath)
        # pad
        trainX_series = trainX_series.apply(lambda x:data_preprocess.pad_sentence(x, x_max_len, vocab_idx_map))
        testX_series = testX_series.apply(lambda x:data_preprocess.pad_sentence(x, x_max_len, vocab_idx_map))
        trainY_series = trainY_series.apply(lambda x:data_preprocess.pad_sentence(x, y_max_len, vocab_idx_map))
        # 保存pad之后的数据
        trainX_series.to_csv(pad_trainX_fpath, index=None, header=False)
        testX_series.to_csv(pad_testX_fpath, index=None, header=False)
        trainY_series.to_csv(pad_trainY_fpath, index=None, header=False)
        # 扩展模型中的词表，并重新训练
        self.wv_model.build_vocab(LineSentence(pad_trainX_fpath), update=True)
        self.wv_model.train(LineSentence(pad_trainX_fpath), 
                        epochs=self.wv_config['wv_train_epochs'], 
                        total_examples=self.wv_model.corpus_count)
        self.wv_model.build_vocab(LineSentence(pad_testX_fpath), update=True)
        self.wv_model.train(LineSentence(pad_testX_fpath), 
                        epochs=self.wv_config['wv_train_epochs'], 
                        total_examples=self.wv_model.corpus_count)
        self.wv_model.build_vocab(LineSentence(pad_trainY_fpath), update=True)
        self.wv_model.train(LineSentence(pad_trainY_fpath), 
                        epochs=self.wv_config['wv_train_epochs'], 
                        total_examples=self.wv_model.corpus_count)


    def wv(self, words):
        '''
        词转向量
        :param words:   [word]  待转的词列表
        :return:        [vec]   词向量列表
        '''
        word_vec_dict = {}
        if 'word2vec' == self.wv_type:
            for word in words:
                if word in self.wv_model:
                    word_vec_dict[word] = list(self.wv_model.wv[word])
        elif 'fasttext' == self.wv_type:
            for word in words:
                word_vec_dict[word] = list(self.wv_model.wv[word])
        return word_vec_dict


    def save_model(self, model_path):
        '''
        保存模型
        :param model_path:  str     模型保存的路径
        :return:    None
        '''
        self.wv_model.save(model_path)


    def load_model(self, model_path):
        '''
        加载模型
        :param model_path:  str     模型的路径
        :return:    None
        '''
        if "fasttext" == self.wv_type:
            self.wv_model = fasttext.FastText.load(model_path)
        elif "word2vec" == self.wv_type:
            self.wv_model = word2vec.Word2Vec.load(model_path)


    def get_embedding_matrix(self, vocab_fpath, save_fpath=None):
        '''
        保存词向量矩阵
        :param vocab_fpath:     str     词表保存地址
        :param save_fpath:      str     词向量保存地址
        :return:    vocabidx_vec_map    {vocab_idx: [embedding_vector}
        '''
        word_idx_map = {}
        with open(vocab_fpath, 'r', encoding='UTF-8') as fp:
            for line in fp:
                items = line.strip('\n').strip('\r').split('\t')
                word = items[0]
                idx = int(items[1])
                word_idx_map[word] = idx
        word_vec_dict = self.wv(word_idx_map.keys())
        idx_vec_map = dict(zip(word_idx_map.values(), word_vec_dict.values()))
        # 保存 embedding matrix
        if save_fpath:
            with open(save_fpath, 'w') as fp:
                for one_idx in sorted(idx_vec_map.keys()):
                    fp.write('\t'.join('{}'.format(one_v) for one_v in [one_idx, idx_vec_map[one_idx]])+'\r\n')
        return idx_vec_map


if __name__ == '__main__':
    # week3
    '''
    wv_config_fpath = './config/wv_config.json'     # 配置文件地址
    wv_type = 'fasttext'                            # 词向量类型
    # 生成对象
    word2vec2_obj = MyWord2Vev2(wv_config_fpath, wv_type)
    # 训练词向量
    merge_seg_data_fpath = './data/processed_data/merge_train_test_seg_data.csv'
    word2vec2_obj.train_wv(merge_seg_data_fpath)
    # 保存词向量模型
    model_path = './model/word2vec/fasttext/fasttext.model'
    word2vec2_obj.save_model(model_path)
    # 构建embedding_matrix并保存
    word2vec2_obj.get_embedding_matrix(vocab_fpath='./data/vocab/vocab.txt',
                                       save_fpath='./data/processed_data/vocabidx_vec_matrix.txt')
    '''

    # week4
    wv_config_fpath = './config/wv_config.json'     # 配置文件地址
    wv_type = 'word2vec'                            # 词向量类型
    # 生成对象
    word2vec2_obj = MyWord2Vev2(wv_config_fpath, wv_type)
    # 加载原先生成的模型（不含start等词）
    model_path = './model/word2vec/word2vec/word2vec.model'
    word2vec2_obj.load_model(model_path)
    # 添加start等特殊字符，重新训练词向量模型
    old_vocab_fpath = './data/vocab/vocab.txt'
    ori_train_fpath = './data/processed_data/train_seg_data.csv'
    ori_test_fpath = './data/processed_data/test_seg_data.csv'
    pad_trainX_fpath = './data/processed_data/trainX_pad.csv'
    pad_trainY_fpath = './data/processed_data/trainY_pad.csv'
    pad_testX_fpath = './data/processed_data/testX_pad.csv'
    word2vec2_obj.expand_vocab(vocab_fpath=old_vocab_fpath, 
                                ori_train_fpath=ori_train_fpath, 
                                ori_test_fpath=ori_test_fpath, 
                                pad_trainX_fpath=pad_trainX_fpath, 
                                pad_trainY_fpath=pad_trainY_fpath, 
                                pad_testX_fpath=pad_testX_fpath)
    # 保存词向量模型
    new_model_path = './model/word2vec/word2vec/word2vec_pad.model'
    word2vec2_obj.save_model(model_path)
    # 构建embedding_matrix并保存
    word2vec2_obj.get_embedding_matrix(vocab_fpath='./data/vocab/vocab_pad.txt',
                                       save_fpath='./data/processed_data/vocabidx_vec_matrix_pad.txt') 
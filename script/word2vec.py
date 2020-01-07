# coding: UTF-8
'''
@author:    wujunbin342@163.com
@date:      2020-01-07
@desc:      word2vec的CBOW和skip-gram实现
'''
import sys
sys.path.append('./script')


class Word2Vec(object):
    '''
    word2vec词向量模型
    '''
    def __init__(self, corpus_fpath, vocab_fpath='./data/vocab/vocab.txt'):
        self.stopwords = set([line.strip() for line in open('./data/vocab/chinese_stopwords.txt', encoding='UTF-8').readlines()])
        self.corpus_fpath = corpus_fpath
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


    def _get_corpus_words_(self):
        '''
        获取语料中的文本（对话文本）,并进行去停用词处理, 切词返回
        :return:
        TODO
        '''


    def get_corpus_by_int(self):
        '''
        获取训练语料，并根据词表转化为int
        :return:
        TODO
        '''
        self.get_vocab()
        words_list = self._get_corpus_words_()   # 获取语料中的文本（对话文本）,并进行去停用词处理
        wordint_list = [self.word_to_int[word] for word in words_list if word not in self.stopwords and word in self.word_to_int]

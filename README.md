# car_master_QA_summary
**汽车大师问答摘要与推理**
[TOC]
## 每周作业
### week1
1. 数据预处理，数据处理与分词，建立词汇表的代码见 `./script/data_preprocess.py`<br>
运行代码使用如下指令：
```bash
>>> python3 script/data_preprocess.py
```
2. 最终生成的词表文件见 `./data/vocab/vocab.txt`。（该文件中词与词index之间的分隔符为`\t`）
### week2
1. **生成词向量训练数据**<br>
生成词向量训练数据在 `script/data_preprocess.py` 文件中，该代码首先生成`DataPreprocess`类的对象`data_preprocess_obj`，然后分别执行`原始数据加载`, `数据处理`, `数据合并与保存`等步骤得到最终的词向量训练数据，并将训练词向量的数据存在`data/processed_data/merge_train_test_seg_data.csv`。<br>
运行代码需修改`script/data_preprocess.py`中`if __name__=="__main__"`选择`week2`的代码，然后执行如下命令：
```bash
>>> python3 script/data_preprocess.py
```
2. **训练词向量，保存词向量模型，并构建embedding_matrix**<br>
训练词向量模型，构建embedding_matrix的代码在`script/word2vec.py`，该代码中依次执行`生成"MyWord2Vec2对象"`, `训练词向量模型`, `保存词向量模型`, `构建embedding_matrix并保存`等步骤, 最终将训练得到的模型保存在`model/word2vec`路径下，将得到的`embedding_matrix`保存在`data/processed_data/vocabidx_vec_matrix.txt`。<br>
运行该代码使用如下命令：
```bash
>>> python3 script/word2vec.py
```
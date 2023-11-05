# 快速开始

### 数据预处理
1. 下载Wizard of Wikipeida 至`./data`数据下，并将其按照两种划分方法做好分类与命名，最后文件夹内容如下
```
data
├── wizard_of_wikipedia_random_split  
│   ├── train.json
│   ├── valid.json 
│   ├── test.json 
│
├── wizard_of_wikipedia_topic_split
│   ├── train.json
│   ├── valid.json 
│   ├── test.json 
│   └── topic_splits.json
```
2. 修改`./data/preprocess_wizard.py`中的输入输出文件参数，运行将两个数据中对应的valid和test数据生成对应的预处理文件（用于计算entity和knowledge F1 Score），最后文件夹内容如下
```
data
├── wizard_of_wikipedia_random_split  
│   ├── train.json
│   ├── valid.json 
│   ├── valid_seen.json 
│   ├── test.json 
│   └── test_seen.json
│
├── wizard_of_wikipedia_topic_split
│   ├── train.json
│   ├── valid.json 
│   ├── valid_seen.json 
│   ├── test.json 
│   ├── test_seen.json
│   └── topic_splits.json
```

### 安装环境
基于ELMER仓库中的环境做出几点修改
#### 1. 更改pytorch版本
```
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```
#### 2. 安装spacy（用于计算entity和knowledge F1 Score）
```
pip install spacy==3.7.0
```
#### 3. 安装spacy中提取entity的模型，两个方法

方法一：网络允许的条件下直接运行
```
python -m spacy download en_core_web_sm
```

方法二：网络条件不允许，则先从[Here](https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0-py3-none-any.whl)下载，然后本地安装

```
pip install [path_to_download_model]
```
#### 4. 安装nltk中相应的包，两个方法
方法一：网络允许的条件下
```
python
>>> import nltk
>>> nltk.download('punkt')
```
方法二：手动下载，先从[Here](https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip)下载，解压后按照如下格式放入目录`~/miniconda3/envs/your_envs_name`下
```
your_envs_name
├── nltk_data
│   ├── tokenizers
│   │   ├── punkt
```

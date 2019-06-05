# PYTORCHでBERTの日本語学習済みモデルを利用する

これはPytorchで日本語の学習済みBERTモデルを読み込み、文章ベクトル(Sentence Embedding)を計算するためのコードです。

詳細は下記ブログを参考ください。

[PytorchでBERTの日本語学習済みモデルを利用する - 文章埋め込み編](https://http://yag-ays.github.io/project/pytorch_bert_japanese/)

## 環境

- 日本語の学習済みBERTモデル: [BERT日本語Pretrainedモデル \- KUROHASHI\-KAWAHARA LAB](http://nlp.ist.i.kyoto-u.ac.jp/index.php?BERT%E6%97%A5%E6%9C%AC%E8%AA%9EPretrained%E3%83%A2%E3%83%87%E3%83%AB)
- BERT実装: [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT)
- 形態素解析器: [JUMAN\+\+ \- KUROHASHI\-KAWAHARA LAB](http://nlp.ist.i.kyoto-u.ac.jp/index.php?JUMAN++)

## 準備

### 日本語の学習済みBERTモデル
京都大学の黒橋・河原研究室が公開している「BERT日本語Pretrainedモデル」を利用します。下記ウェブページからモデルファイルをダウンロードして解凍します。

[BERT日本語Pretrainedモデル \- KUROHASHI\-KAWAHARA LAB](http://nlp.ist.i.kyoto-u.ac.jp/index.php?BERT%E6%97%A5%E6%9C%AC%E8%AA%9EPretrained%E3%83%A2%E3%83%87%E3%83%AB)

### Juman++
Juman++をインストールします。インストール方法については、下記の公式レポジトリを参照ください。

[ku\-nlp/jumanpp: Juman\+\+ \(a Morphological Analyzer Toolkit\)](https://github.com/ku-nlp/jumanpp)

macOSならば、Homebrewを使って下記のようにインストールできます。

```sh
$ brew install jumanpp
```

### Pythonパッケージ
`pytorch-pretrained-bert`および`pyknp`をインストールします。

```sh
$ pip install pytorch-pretrained-bert
$ pip install pyknp
```

## 実行する
本レポジトリの`bert_juman`から`BertWithJumanModel`クラスをインポートします。クラスの引数には、ダウンロードした日本語の学習済みBERTモデルのディレクトリを指定します。


```py
In []: from bert_juman import BertWithJumanModel

In []: bert = BertWithJumanModel("/path/to/Japanese_L-12_H-768_A-12_E-30_BPE")

In []: bert.get_sentence_embedding("吾輩は猫である。")
Out[]:
array([ 2.22642735e-01, -2.40221739e-01,  1.09303640e-02, -1.02307117e+00,
        1.78834641e+00, -2.73566216e-01, -1.57942638e-01, -7.98571169e-01,
       -2.77438164e-02, -8.05811465e-01,  3.46736580e-01, -7.20409870e-01,
        1.03382647e-01, -5.33944130e-01, -3.25344890e-01, -1.02880754e-01,
        2.26500735e-01, -8.97880018e-01,  2.52314955e-01, -7.09809303e-01,
[...]        
```

また`get_sentence_embedding()`の引数には、文章ベクトルを計算するのに利用するBERTの隠れ層の位置`pooling_layer`と、プーリングの方法`pooling_strategy`が指定できます。`pooling_layer`は`-1`で最終層、`-2`で最終層の手前の層となります。また、`pooling_strategy`には

- `REDUCE_MEAN`: 要素ごとにaverage-pooling
- `REDUCE_MAX`: 要素ごとにmax-pooling
- `REDUCE_MEAN_MAX`: `REDUCE_MEAN`と`REDUCE_MAX`を結合したもの

が選択できます。

```py
In []: bert.get_sentence_embedding("吾輩は猫である。",
   ...:                             pooling_layer=-1,
   ...:                             pooling_strategy="REDUCE_MAX")
   ...:
Out[]:
array([ 1.2089624 ,  0.6267309 ,  0.7243419 , -0.12712255,  1.8050476 ,
        0.43929055,  0.605848  ,  0.5058241 ,  0.8335829 , -0.26000524,
[...]        
```

これらのパラメータは[hanxiao/bert-as-service](https://github.com/hanxiao/bert-as-service)を参考にしています。

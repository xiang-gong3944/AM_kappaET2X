
# 概要

中先生の交代磁性の論文Spin current generation in organic antiferromagnets
[リンク](https://www.nature.com/articles/s41467-019-12229-y)
と「交替磁性体のスピン分裂と交差相」(固体物理2024-05)の中に出てくる
κ-(ET)2-X についての計算についてのフォルダである。

# 構造

フォルダの構造は以下のとおりである。

``` sh
AM_kappaET2X
│  kappaET2X.ipynb      # 簡単なノート
│  readme.md            # これ
│  run.ipynb            # 実行テンプレート
│
├─image         # PowerPoint 等で策表する図表
│
├─model
│  │  calc.py                   # 物理量の計算
│  │  hubbard_model.py          # 平均場ハバード模型の自己無撞着計算と変数
│  │  operators.py              # 平均場ハミルトニアンと流れ演算子
│  │  plot.py                   # 計算結果のグラフの出力
│  │  __init__.py
│
├─output            # 計算結果の図の保存先
│
├─util
│  │  post.py           # Discord への通知
│  │  __init__.py
│
└─資料          # 関連しそうな文献
```

# 使い方

こんな感じで計算できる

``` python
CuO2 = HubbardModel(n_carrier=6.1, k_mesh=75, U=1, iteration=400, err=1e-6, fineness=5)
calc.spin_conductivity(KappaET2X, "x", "y")
plot.fermi_surface(KappaET2X)
plot.dos(KappaET2X)
plot.nsite(KappaET2X)
plot.scf(KappaET2X)
plot.band(KappaET2X)
```

# その他

`util/` の中には `__init__.py` と `post.py` が入っている。

`__init__.py` はモジュールとしてこのフォルダを読み取るためのもので、
まぁよくあるやつである。

`post.py` は Discord や Slack に webhook を通じて通知をするものである。
これを使うことで長い計算の進捗をスマホで見ることができるようになる。

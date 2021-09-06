# Transfer Learning Using Diff Pruning
Sources:
 - [[Diff Pruning 2020 Paper]](https://arxiv.org/abs/2012.07463) [[Repository]](https://github.com/dguo98/DiffPruning)
 - [[BioBERT 2019 Paper]](https://academic.oup.com/bioinformatics/article/36/4/1234/5566506)

## Results

**Task 1**: finetuning [BioBERT](https://github.com/dmis-lab/biobert-pytorch) with diff-vector on [BC2GM](https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data/BC2GM-IOBES)

  - Precision:   85.56
  - Recall:   87.02
  - F1: 86.28

 

Parameter  | Value
------------ | -------------
epochs       | 80
w learning rate| 1e-5
alpha learning rate| 1e-5
weight decay | 1e-2
batch size   | 32



**Task 2**: diff pruning [BioBERT](https://github.com/dmis-lab/biobert-pytorch) with diff-vector on [BC2GM](https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data/BC2GM-IOBES)

Parameter  | Value
------------ | -------------
epochs       | 80
w learning rate| 1e-5
alpha learning rate| 1e-1
sparsity penalty | 1.25e-7
weight decay | 1e-2
batch size   | 32

  - Nonzero parameters: 1.1%
  - Precision:   84.73
  - Recall:   83.32
  - F1: 86.19

Magnitude Pruning and Fixmask Finetuning:

Parameter  | Value
------------ | -------------
epochs       | 80
w learning rate| 1e-5
alpha learning rate| -
sparsity penalty | -
weight decay | 1e-2
batch size   | 32

  - Nonzero parameters:0.5%
  - Precision:   86.43
  - Recall:   84.33
  - F1: 85.37

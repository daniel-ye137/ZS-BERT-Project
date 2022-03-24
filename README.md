# ZS-BERT
This repository contains the implementation of the NAACL 2021 paper "Towards Zero-Shot Relation Extraction with Attribute Representation Learning".

# Dataset
You can download the datasets employed in our work from the following link:
- [WikiZSL (Daniil Sorokin and Iryna Gurevych, 2017)](https://drive.google.com/file/d/1ELFGUIYDClmh9GrEHjFYoE_VI1t2a5nK/view?usp=sharing)
- [FewRel (Xu et al., 2018)](https://drive.google.com/file/d/1QY-5R2zqLPnT5DDF5phTbSvEeBkQRh7A/view?usp=sharing)

and place them to the `/data` folder.

# Structure
```
ZS-BERT/
├── model
    ├── model.py
    ├── data_helper.py
    ├── evaluation.py
    ├── train_wiki.py
    └── train_fewrel.py
└── resources/
    ├── property_list.html
└── data/
    ├── wiki_train_new.json
    └── fewrel_all.json
```

# Requirements
python >= 3.6
torch >= 1.4.0
or simply run:
```
pip install -r requirements.txt
```

# Train ZS-BERT
If you wish to train on the wiki dataset, run:
```
python3 train_wiki.py --seed 300 --n_unseen 10 --gamma 7.5 --alpha 0.4 --dist_func 'inner' --batch_size 4 --epochs 10
```
Otherwise to train on FewRel dataset, you can run:
```
python3 train_fewrel.py --seed 300 --n_unseen 10 --gamma 7.5 --alpha 0.4 --dist_func 'inner' --batch_size 4 --epochs 10
```
inside the `/model` folder.

# Citing this paper
If you use the code, we appreciate it if you cite the following paper:

```
@inproceedings{chen2021zsbert,
  title={ZS-BERT: Towards Zero-Shot Relation Extraction with Attribute Representation Learning},
  author={Chih-Yao Chen and Cheng-Te Li},
  booktitle={Proceedings of 2021 Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL-2021)},
  year={2021}
}
```

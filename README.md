# Representation Learning for Entity Alignment in Knowledge Graph: A Design Space Exploration

## Dependencies
- python: 3.6.10
- pytorch: 1.7.1
- transformers: 3.0.0
- tqdm: 4.64.1

## Installation
We recommend creating a new conda environment to install the dependencies and run SDEA:

```shell
conda create -n SDEA python=3.6.10
conda activate SDEA
conda install pytorch-gpu=1.7.1
pip install transformers==3.0.0
```

## Preparation

The structure of the project is listed as follows:

```
Dual_strategy/
├── src/: The soruce code of SDEA. 
├── data/: The datasets. 
│   ├── DBP15k/: The downloaded DBP15K benchmark. 
│   │   ├── fr_en/
│   │   ├── ja_en/
│   │   ├── zh_en/
│   ├── DWY100K/: The downloaded OpenEA benchmark. 
│   │   ├── dbp_yg_dwy/
│   │   ├── dbp_wd_dwy/
│   ├── SRPRS/: The downloaded SRPRS benchmark. 
│   │   ├── en_de_15k_V1/
│   │   ├── en_fr_15k_V1/
│   │   ├── dbp_wd_15k_V1/
│   │   ├── dbp_yg_15k_V1/
│   ├── OpenEA/: The downloaded OpenEA benchmark. 
│   │   ├── EN_DE_15K_V1/
│   │   ├── EN_FR_15K_V1/
│   │   ├── D_W_15K_V1/
│   │   ├── D_Y_15K_V1/
│   │   ├── EN_DE_100K_V1/
│   │   ├── EN_FR_100K_V1/
│   │   ├── D_W_100K_V1/
│   │   ├── D_Y_100K_V1/
│   ├── DBP15k_no/: DBP15K without aligned neighbor. 
│   │   ├── fr_en/
│   │   ├── ja_en/
│   │   ├── zh_en/
│   │   │   ├── train_links :Training set (id_1,id_2) 
│   │   │   ├── valid_links 
│   │   │   ├── test_links
│   │   │   ├── ent_ids_1 :(id,entity name)
│   │   │   ├── ent_ids_2
│   │   │   ├── ent_links_id :all pairs (id_1,id_2)
│   │   │   ├── comment_1 :(id,description)
│   │   │   ├── comment_2
│   │   │   ├── rel_ids_1 :(id,relation name)
│   │   │   ├── rel_ids_2
│   │   │   ├── triples_1 :(head entity id,relation id,tial entity id)
│   │   │   ├── triples_2   
│   │   │   ├── attr_triples_1/2attr :(entity name,Attribute name,Value name)
│   │   │   ├── attr_triples_2/2attr      
├── bert-base/: The pre-trained transformer-based models. 
│   ├── bert-base-multilingual-cased: The model used in our experiments.
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   ├── vocab.txt
├── save_data/: save intermediate data
│   ├── DBP15k/:
│   │   ├── fr_en/
│   │   ├── ja_en/
│   │   ├── zh_en/
│   ├── ......
```

### Datasets

- SRPRS: https://github.com/nju-websoft/RSN/raw/master/entity-alignment-full-data.7z

- DBP15K:  Please download from [Google Drive](https://drive.google.com/file/d/1Xj6CaeECTDwuJM5nj_Xk3JZt_oXFu5sO/view?usp=sharing).
- OpenEA: https://github.com/nju-websoft/OpenEA
- DWY100K: https://github.com/THU-KEG/KECG
1. Download the datasets and unzip them into "Dual-strategy/data".
2. Preprocess the datasets.
- DBP15k_no: copy DBP15k and run del_neb.py
```
cd data
python del_neb.py
```
### Synthesized Datasets
|  dataset   | origin  | operation  |
|  ----  | ----  | ----  |
| 1  | DBP15k | remove relation,attribute
| 2  |  DBP15k | remove attribute
| 3  |  DBP15k_no | remove textual information,attribute
| 4  |  DBP15k_no | remove attribute
| 5  | DBP15k | 
| 6  |  DBP15k_no | 
| 7  |  DBP15k | remove textual information
| 8  |  DBP15k_no | remove textual information
### Pre-trained Models

The pre-trained models of transformers library can be downloaded from https://huggingface.co/models. 
We use [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased) in our experiments. 

Please put the downloaded pre-trained models into "Dual_strategy/bert-base". 


## How to Run
Embedding initialization:
```
cd src/Initialization
textual information: model=bert_int_des,data_type=entity
python main.py
textual information: model=bert_int_des,data_type=entity
python main.py
relation: model=bert_int_des,data_type=relation
python main.py
attribute: model=bert_int_des,data_type=attribute
python main.py
```
Attribute encoding:
```
cd src/Initialization
python main.py :model=SDEA,data_type=entity
python main.py :model=SDEA,data_type=relation
```
Structural information encoding:
Attribute/config: choose "config_args_RREA_literal" row line:606
```
cd src/Attribute  
python main.py
```
Attribute interaction:
Attribute/config: choose "config_args_BERT_INT_A" row line:606
```
cd src/Attribute  
python main.py
```

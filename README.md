# Style Knowledge Graph: Augmenting Text Style Transfer with Knowledge Graphs

This is the official repository for the paper [Style Knowledge Graph: Augmenting Text Style 
Transfer with Knowledge Graphs](https://aclanthology.org/2025.genaik-1.13/).

Text style transfer is the task of modifying the stylistic attributes of a given text while 
preserving its original meaning. This task has also gained interest with the advent of large 
language models. Although knowledge graph augmentation has been explored in various tasks, its 
potential for enhancing text style transfer has received limited attention. This paper proposes 
a method to create a Style Knowledge Graph (SKG) to facilitate and improve text style transfer. 
The SKG captures words, their attributes, and relations in a particular style, that serves as a 
knowledge resource to augment text style transfer. We conduct baseline experiments to evaluate 
the effectiveness of the SKG for augmenting text style transfer by incorporating relevant parts 
from the SKG in the prompt. The preliminary results demonstrate its potential for enhancing 
content preservation and style transfer strength in text style transfer tasks, while the results 
on fluency indicate promising outcomes with some room for improvement. We hope that the proposed 
SKG and the initial experiments will inspire further research in the field. 

Read the full paper [here](https://aclanthology.org/2025.genaik-1.pdf#page=133). If you find 
this paper useful or use some of our created SKGs, please kindly cite our paper:

```
@inproceedings{toshevska2025style,
  title={Style Knowledge Graph: Augmenting Text Style Transfer with Knowledge Graphs},
  author={Toshevska, Martina and Kalajdziski, Slobodan and Gievska, Sonja},
  booktitle={Proceedings of the Workshop on Generative AI and Knowledge Graphs (GenAIK)},
  pages={123--135},
  year={2025}
}
```

The main contributions of the paper are: 
- We propose a knowledge graph for text style transfer, which we refer to as a *Style 
  Knowledge Graph (SKG)*. 
- We evaluate the effectiveness of augmenting text style transfer with SKG via prompting.
- We analyze the influence of various parts of the SKG on the text style transfer task.

To get access to the created style knowledge graphs, feel free to contact the authors via email 
```martina.toshevska@finki.ukim.mk```.

A visual representation of the proposed approach is shown in the following figure:
![](images/Style_knowledge_graph.png)

The architecture of the proposed prompting methods is shown in the following figure:
![](images/SKG_prompting_architecture.png)

## Links to access the created style knowledge graphs

- **Yelp**
  - [Original dataset](https://arxiv.org/pdf/1605.05362)
  - [SKG](https://finkiukim-my.sharepoint.com/:f:/g/personal/martina_toshevska_finki_ukim_mk/EuAJBvhkl3FMhi1EmnaJSsQB2vUpJXd9Xc4E3gCc1Wxa7g?e=Sx20yx)
- **Politeness**
  - [Original dataset](https://aclanthology.org/2020.acl-main.169.pdf)
  - [SKG](https://finkiukim-my.sharepoint.com/:f:/g/personal/martina_toshevska_finki_ukim_mk/EtEgfx13eotOrgjhTRlmOcgBAjC4Lmf3Bz_Gn7BrI7jUkQ?e=OafZFx)
- **GYAFC**
  - [Original dataset](https://aclanthology.org/N18-1012.pdf)
  - To access the *SKG* for the GYAFC dataset please sent an email with the confirmation about 
    being granted the access to the original GYAFC dataset to ```martina.toshevska@finki.ukim.
    mk```. After that you will receive a private link to access the SKG.
- **WNC**
  - [Original dataset](https://arxiv.org/pdf/1911.09709)
  - [SKG](https://finkiukim-my.sharepoint.com/:f:/g/personal/martina_toshevska_finki_ukim_mk/ElNLAqeBuclOuq0bwkj16XIBX2QAa1sgg3ggAiQjZPV9IQ?e=E5MNy2)
- **Shakespeare**
  - [Original dataset](https://aclanthology.org/C12-1177.pdf)
  - [SKG](https://finkiukim-my.sharepoint.com/:f:/g/personal/martina_toshevska_finki_ukim_mk/EvT9hrdnL9lIno85yDI1Bj4B68o3AEJqLVj0ZgFnlVzseg?e=NqavkR)
- **ParaDetox**
  - [Original dataset](https://aclanthology.org/2022.acl-long.469.pdf)
  - [SKG](https://finkiukim-my.sharepoint.com/:f:/g/personal/martina_toshevska_finki_ukim_mk/Enm6jBfPMtRKt4cM2ccwMkABRdjHxtJgQH1xjZWS42RXIw?e=RblsiV)

## Creating you own SKG

**Step 1**: prepare you dataset in the format as the provided sample in the 
```sample_data/non-parallel``` directory.

**Step 2**: run the ```create_graph.py``` script as follows
```shell
cd graph_creation

python create_graph.py --dataset_name <dataset_name> --style_1_name <style_1_name> 
--style_2_name <style_2_name>
```

**(Optional) Step 3**: to print the statistics for your created SKG run the following
```shell
cd graph_creation

python statistics.py --dataset_name <dataset_name> --style_1_name <style_1_name> 
--style_2_name <style_2_name>
```

**(Optional) Step 4**: to create the triples for your created SKG run the following
```shell
cd graph_creation

python prepare_triples.py --dataset_name <dataset_name> --style_1_name <style_1_name> 
--style_2_name <style_2_name>
```

## Example commands for running SKG-augmented prompting

First, prepare you dataset in the format as the provided sample in the 
```sample_data/parallel``` directory.

```shell
cd zero_shot_skg_prompting

python predict_t5.py --dataset_name <dataset_name> --model_name <model_name> --dataset_types 
test --batch_size 128 --n_few_shot 0

python predict_llama.py --dataset_name <dataset_name> --model_name <model_name> --dataset_types 
test --batch_size 128 --n_few_shot 0
```

```shell
cd few_shot_skg_prompting

python predict_t5.py --dataset_name <dataset_name> --model_name <model_name> --dataset_types 
test --batch_size 128 --n_few_shot 2

python predict_llama.py --dataset_name <dataset_name> --model_name <model_name> --dataset_types 
test --batch_size 128 --n_few_shot 2
```
# bert for docred
## docred 
Dataset and code for baselines for [DocRED: A Large-Scale Document-Level Relation Extraction Dataset](https://arxiv.org/abs/1906.06127v3)

Multiple entities in a document generally exhibit complex inter-sentence relations, and cannot be well handled by existing relation extraction (RE) methods that typically focus on extracting intra-sentence relations for single entity pairs. In order to accelerate the research on document-level RE, we introduce DocRED, a new dataset constructed from Wikipedia and Wikidata with three features: 

+ DocRED annotates both named entities and relations, and is the largest human-annotated dataset for document-level RE from plain text.
+ DocRED requires reading multiple sentences in a document to extract entities and infer their relations by synthesizing all information of the document.
+ Along with the human-annotated data, we also offer large-scale distantly supervised data, which enables DocRED to be adopted for both supervised and weakly supervised scenarios.
## use bert to do docred
1. hange the bert path and the data path in main.py to your own path.
2. 
```
    cd tf_version
    python main.py
```
3. You will see the following training information
![image](https://github.com/zhongerqiandan/bert4docred/blob/master/images/WechatIMG8.png)
4. [Leaderboard](https://competitions.codalab.org/competitions/20717)
The score of our model in the test set are roughly as follows.
```
re_ign_ann_f1 = 0.574
re_ign_dis_f1 = 0.535
```
5. Keep updating,if I'm in a good mood.
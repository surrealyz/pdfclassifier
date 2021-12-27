# On Training Robust PDF Malware Classifiers

Code for our paper [On Training Robust PDF Malware Classifiers](https://arxiv.org/abs/1904.03542) (Usenix Security'20)
Yizheng Chen, Shiqi Wang, Dongdong She, Suman Jana

## Blog Posts

[Monotonic Malware Classifiers (5 min read)](https://surrealyz.medium.com/monotonic-malware-classifiers-83cd4451f58d)

[Gmail's malicious document classifier can still be trivially evaded (3 min read)](https://surrealyz.medium.com/gmails-malicious-document-classifier-can-still-be-trivially-evaded-93e625745c9d)

## Dataset

#### Full PDF dataset

Available [here at contagio](http://contagiodump.blogspot.com/2013/03/16800-clean-and-11960-malicious-files.html): "16,800 clean and 11,960 malicious files for signature testing and research."

#### Training and Testing datasets

We split the PDFs into 70% train and 30% test. Then, we used the [Hidost feature extractor](https://github.com/srndic/hidost) to
extract structural paths features, with the default `compact` path option.
We obtained the following training and testing data.

|   | Training PDFs  | Testing PDFs  |
|---|---|---|
| Malicious | 6,896 | 3,448 |
| Benign | 6,294 | 2,698 |

The hidost structural paths are [here](https://github.com/surrealyz/pdfclassifier/tree/master/data/extracted_structural_paths).

The extracted training and testing libsvm files are [here](https://github.com/surrealyz/pdfclassifier/tree/master/data/traintest_all_500test). The 500 seed malware samples with network activities from EvadeML are in the test set.

[500 seed malware hash list.](https://github.com/surrealyz/pdfclassifier/blob/master/data/seeds_hash_list.txt). Put these PDFs under `data/500_seed_pdfs/`.

## Models

The following models are TensorFlow checkpoints, except that two ensemble models need additional wrappers.

| Checkpoint |  Model |
|---|---|
| baseline_checkpoint  | Baseline  |
| baseline_adv_delete_one  | Adv Retrain A  |
| baseline_adv_insert_one  | Adv Retrain B  |
| baseline_adv_delete_two  | Adv Retrain C  |
| baseline_adv_insert_rootallbutone  | 	Adv Retrain D  |
| baseline_adv_combine_two  | Adv Retrain A+B  |
| adv_del_twocls  | Ensemble A+B Base Learner  |
| adv_keep_twocls  | Ensemble D Base Learner  |
| robust_delete_one  | Robust A  |
| robust_insert_one  | Robust B  |
| robust_delete_two  | Robust C  |
| robust_insert_allbutone  | Robust D  |
| robust_monotonic  | Robust E  |
| robust_combine_two_v2_e18  | Robust A+B  |
| robust_combine_three_e17  | Robust A+B+E  |

The following are XGBoost tree ensemble models.

| Binary  | Model  |
|---|---|
| model_10learner_test.bin  | Monotonic Classifier, 10 learners  |
| model_100learner.bin  | Monotonic Classifier, 100 learners  |
| model_1000learner.bin  | Monotonic Classifier, 1000 learners  |
| model_2000learner.bin  | 	Monotonic Classifier, 2000 learners  |

## Training Code

To train and evaluate the VRAs of baseline model, adv retrain models, ensemble models, XGBoost monotonic models, and robust models, see [README](https://github.com/surrealyz/pdfclassifier/tree/master/train) under `train/`.

## Attacks in the Paper

See [README](https://github.com/surrealyz/pdfclassifier/tree/master/attack) under `attack/`.

<p align="center">
  <img src="https://surrealyz.github.io/image/era_feat_adaptive.png" alt="robust_gbdt" width="50%" height="50%"/>
</p>

After running our adaptive attack based on EvadeML against the Robust A+B+E model
for three weeks, we were not able to fully evade the model to generate functional evasive
PDF malware variants. As shown in the Figure above, the estimated robust accuracy
against adaptive attacks can be reduced to 0% for Monotonic 100 and Robust A+B models,
but not Robust A+B+E model. We hope researchers can design stronger attacks
to evade our Robust A+B+E model.

Using the [EvadeML framework](https://github.com/surrealyz/EvadeML), our adaptive strategies against
this model are:
* **Move Exploit Attack.** The monotonic property (Property E) forces the
attacker to delete objects from the malware, but deletion could
remove the exploit. Therefore, we implement a new mutation
to move the exploit around to different trigger points in the PDF.
* **Scatter Attack.** To evade Robust A+B and Robust A+B+E, we insert and delete
more objects under different subtrees. We keep track of past
insertion and deletion operations separately, and prioritize new
insertion and deletion operations to target a different subtree.

## MalGAN Attack Evaluation

Please check out this [MalGAN attack evaluation](https://github.com/xiaoluLucy814/Malware-GAN-attack) against our robust models by [Zeyi](https://github.com/xiaoluLucy814/).

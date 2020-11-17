## Bounded Arbitrary Attacker

See [README](https://github.com/surrealyz/pdfclassifier/tree/master/train) under `../train/` on evaluating Verified Robust Accuracy for the models.

## Gradient Attackers

For example,
`python gradient_attack.py --method un --model baseline` runs unrestricted gradient attack against the baseline NN model.

The following attack methods are available:

| --method option | Attack Config |
|---|---|
| un | unrestrcited gradient attack|
| uni | unrestricted insertion-only attack (Property E) |
| und | unrestricted deletion-only attack |
| A | restricted gradeint attack bounded by Property A |
| B | restricted gradeint attack bounded by Property B |
| C | restricted gradeint attack bounded by Property C |
| D | restricted gradeint attack bounded by Property D |

The following model name options correspond to the neural network models:

| --model option | Checkpoint |  Model |
|---|---|---|
| baseline | baseline_checkpoint  | Baseline  |
| TA | baseline_adv_delete_one  | Adv Retrain A  |
| TB | baseline_adv_insert_one  | Adv Retrain B  |
| TC | baseline_adv_delete_two  | Adv Retrain C  |
| TD | baseline_adv_insert_rootallbutone  | 	Adv Retrain D  |
| ATAB | baseline_adv_combine_two  | Adv Retrain A+B  |
| EAB | adv_del_twocls  | Ensemble A+B Base Learner  |
| ED | adv_keep_twocls  | Ensemble D Base Learner  |
| RA | robust_delete_one  | Robust A  |
| RB | robust_insert_one  | Robust B  |
| RC | robust_delete_two  | Robust C  |
| RD | robust_insert_allbutone  | Robust D  |
| mono | robust_monotonic  | Robust E  |
| RAB | robust_combine_two_v2_e18  | Robust A+B  |
| RABE | robust_combine_three_e17  | Robust A+B+E  |


### Pack the evasive feature vectors back to real PDFs

Put the benign PDF dataset under here `../data/traintest_all_500test/train_benign`.
Download the feature vectors resulting from the unrestricted gradient attacks from [here](https://drive.google.com/file/d/1zHT_Pm27EbO7IsLAOxSex0lwLCu_12yC/view?usp=sharing). Extract the file and put it under `../data/`. Put the 500 seed PDFs with network signatures under `../data/500_seed_pdfs/`.

Run `python gradient_pdf_packback.py`.

Given each feature index change, we either delete the corresponding PDF object, or insert the object with minimal number of children in the benign training dataset. Inserting object with minimal children makes the features from constructed PDF close to the evasive features. On average, the ERA of models against the real evasive PDF malware is 94.25%, much higher than 0.62% ERA against evasive feature vectors, since unrestricted gradient attack often breaks the PDF semantics.

## MILP Attacker

The Mixed Integer Linear Program (MILP) attack was proposed in "Evasion and Hardening of Tree Ensemble Classifiers" by Alex Kantchelian, J. D. Tygar, and Anthony D. Joseph, ICML 2016.

The attack script is modified from the one under [RobustTrees](https://github.com/chenhongge/RobustTrees/blob/master/xgbKantchelianAttack.py). You will need to install Gurobi solver and obtain an academic license to run it.

The following example minimize the L<sub>0</sub> distance for the MILP Attacker, and targets the model `model_100learner`. You can modify the attack objective and the model in the command.
```
o='0'; md='model_100learner'; r="_l${o}"; python xgbKantchelianAttack.py --order ${o} --data '../data/traintest_all_500test/test_data.libsvm' --model_type 'xgboost' --model "../models/monotonic/${md}.bin" --num_classes 2 --nfeat 3514 --maxone --feature_start 0 --out "milp_${md}${r}.txt" --adv "adv_examples_${md}${r}.pickle" > log
```


## Enhanced Evolutionary Attacker


## Reverse Mimicry Attacker


## Adaptive Evolutionary Attacker

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
| A | restricted gradient attack bounded by Property A |
| B | restricted gradient attack bounded by Property B |
| C | restricted gradient attack bounded by Property C |
| D | restricted gradient attack bounded by Property D |

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

The Mixed Integer Linear Program (MILP) attack was proposed in "Evasion and Hardening of Tree Ensemble Classifiers" by Alex Kantchelian, J. D. Tygar, and Anthony D. Joseph. ICML 2016.

The attack script is modified from the one under [RobustTrees](https://github.com/chenhongge/RobustTrees/blob/master/xgbKantchelianAttack.py). You will need to install the Gurobi solver and obtain a license to run it. They provide free academic license.

The following example minimize the L<sub>0</sub> distance for the MILP Attacker, and targets the model `model_100learner`. You can modify the attack objective and the model in the command.
```
o='0'; md='model_100learner'; r="_l${o}"; python xgbKantchelianAttack.py --order ${o} --data '../data/traintest_all_500test/test_data.libsvm' --model_type 'xgboost' --model "../models/monotonic/${md}.bin" --num_classes 2 --nfeat 3514 --maxone --feature_start 0 --out "milp_${md}${r}.txt" --adv "adv_examples_${md}${r}.pickle" > log
```

## Reverse Mimicry Attacker

We implement our own reverse mimicry attack, similar to the
JSinject. We use [peepdf](https://github.com/jesparza/peepdf) tatic analyzer to identify the
suspicious objects in the PDF malware seeds, and then inject
these objects to a benign PDF. We inject different malicious
payload into a benign file, whereas the JSinject attack injects
the same JavaScript code into different benign PDFs. Within
the PDF malware seeds, 250 of them retained maliciousness
according to the cuckoo oracle. Some payload are no longer
malicious because there can be object dependencies within the
malware not identified by the static analyzer. We test whether
the models can detect the 250 PDFs are malicious.

The following generates reverse mimicry PDFs by inserting malicious payload
from the seeds to the benign PDF `win08.pdf`.
```
cd reverse_mimicry;
mkdir test_files;
python all_insert_epath.py
```

We also provided a script `all_check_cuckoo.py` that we used to check whether
the cuckoo oracle thinks the generated PDFs are malicious, with the results
in `cuckoo_test_files.tsv`. You can reference the script, but will need to change
it according to your cuckoo and evademl setup.
Among these that were verified to be malicious by the cuckoo oracle,
different classifiers have different accuracy for them, see Table 6 in our paper.

JSInject was proposed in "Looking at the bag is not enough to find the bomb: an evasion of structural methods for malicious PDF files detection". Davide Maiorca, Igino Corona, and Giorgio Giacinto. ASIA CCS 2013.

## Enhanced and Adaptive Evolutionary Attacker

Fork [the original EvadeML](https://github.com/uvasrg/EvadeML) if you would like to experiment from the official version.
To reproduce the results in our paper, fork [my version of EvadeML](https://github.com/surrealyz/EvadeML) that included the necessary enhancements and adaptive attacks.

You will first need to follow all the necessary setups according to the
documentation of EvadeML.

Then, the following are the scripts to attack different models **under my EvadeML fork**.
A lot of paths will need to be modified according to your setup, e.g.,
the home directory `/home/yz` needs to be replaced by yours.

### Attack the baseline NN
`run_regular.sh`

### Attack Adv Retrain A+B
`run_baseline_adv.sh`. The classifer wrapper is `classifiers/baseline_adv_wrapper.py`.

### Attack the Ensemble A+B model
`run_ensemble.sh`. The ensemble classifer wrapper is `classifiers/ensemble_wrapper.py`.

### Attack the Robust A+B model
`run_robust_combine_two.sh`. The classifier wrapper is `classifiers/robustmlp_wrapper.py`.

### Adaptive Attack against the monotonic classifier
`run_monotonic_deletion.sh` deletion only attack can evade about 50% of PDFs.
Then use the script `batch_monotone_reuse_adaptive.py` to do adaptive attack
that includes a `move` mutation operation.

### Adaptive Attack against the Robust A+B model
`run_robust_combine_two_adaptive.sh`

### Adaptive Attack against the Robust A+B+E model
`run_robust_combine_three_adaptive.sh`

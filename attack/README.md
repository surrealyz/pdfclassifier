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
Download the feature vectors resulting from the unrestricted gradient attacks from [here](https://drive.google.com/file/d/1zHT_Pm27EbO7IsLAOxSex0lwLCu_12yC/view?usp=sharing). Extract the file and put it under `../data/`. Put the 500 seed PDFs with network signatures under `../data/500_seeds/`.

Run `python gradient_pdf_packback.py`.

Given each feature index change, we either delete the corresponding PDF object, or insert the object with minimal number of children in the benign training dataset. Inserting object with minimal children makes the features from constructed PDF close to the evasive features. On average, the ERA of models against the real evasive PDF malware is 94.25%, much higher than 0.62% ERA against evasive feature vectors, since unrestricted gradient attack often breaks the PDF semantics.

## MILP Attacker


## Enhanced Evolutionary Attacker


## Reverse Mimicry Attacker


## Adaptive Evolutionary Attacker

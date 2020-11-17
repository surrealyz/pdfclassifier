## Neural Network Models

### Training Scripts

The mini-batch numbers are calculated to be 20 epochs below.

#### Interval Datasets
Please download the interval datasets from [this link](https://drive.google.com/file/d/11xPFBfxpaU3YHSI0W46yST2SJD4luz79/view?usp=sharing), and extract it under the `robustness_spec/` directory.


#### Baseline NN
* Train: `python train.py --baseline --batches 5276`

#### Adv Retrain A
* Train:
```
time python train_adv.py --train robustness_spec/seed_train_malicious/mutate_delete_one/pickles --batch_size 50 --batches 17520 --verbose 1000 --test robustness_spec/seed_test_malicious/mutate_delete_one/pickles --test_batches 315 --model_name baseline_adv_delete_one
```

#### Adv Retrain B
* Train:
```
time python train_adv_ins.py --train robustness_spec/seed_train_malicious/mutate_insert_any_pt1/pickles --batch_size 50 --batches 31450 --verbose 2000 --test robustness_spec/seed_test_malicious/mutate_insert_rootany/pickles --test_batches 2869 --model_name baseline_adv_insert_one
```
```
time python train_adv_ins.py --resume --train robustness_spec/seed_train_malicious/mutate_insert_any_pt2/pickles --batch_size 50 --batches 31500 --verbose 2000 --test/robustness_spec/seed_test_malicious/mutate_insert_rootany/pickles --test_batches 2869 --model_name baseline_adv_insert_one
```

#### Adv Retrain C
* Train:
```
time python train_adv.py --train robustness_spec/seed_train_malicious/mutate_delete_two/pickles --batch_size 50 --batches 30240 --verbose 2000 --test robustness_spec/seed_test_malicious/mutate_delete_two/pickles --test_batches 682 --model_name baseline_adv_delete_two
```

#### Adv Retrain D
* Train:
```
time python train_adv_ins.py --train robustness_spec/seed_train_malicious/mutate_insert_rootallbutone_pt1/pickles --batch_size 50 --batches 8520 --verbose 2000 --test
robustness_spec/seed_test_malicious/mutate_insert_rootallbutone/pickles --test_batches 2869 --model_name baseline_adv_insert_rootallbutone
```
```
time python train_adv_ins.py --resume --train robustness_spec/seed_train_malicious/mutate_insert_rootallbutone_pt2/pickles --batch_size 50 --batches 8520 --verbose 2000 --test robustness_spec/seed_test_malicious/mutate_insert_rootallbutone/pickles --test_batches 2869 --model_name baseline_adv_insert_rootallbutone; python train_adv_ins.py --resume --train robustness_spec/seed_train_malicious/mutate_insert_rootallbutone_pt3/pickles --batch_size 50 --batches 8525 --verbose 2000 --test robustness_spec/seed_test_malicious/mutate_insert_rootallbutone/pickles --test_batches 2869 --model_name baseline_adv_insert_rootallbutone; python train_adv_ins.py --resume --train robustness_spec/seed_train_malicious/mutate_insert_rootallbutone_pt4/pickles --batch_size 50 --batches 8545 --verbose 2000 --test robustness_spec/seed_test_malicious/mutate_insert_rootallbutone/pickles --test_batches 2869 --model_name baseline_adv_insert_rootallbutone;
```

#### Adv Retrain A+B
* Train:
```
time python train_adv_combine.py --batch_size 50 --batches 132900 --verbose 2000 --model_name baseline_adv_combine_two
```

#### Robust A

* Train:
```
time python train_delete.py --train robustness_spec/seed_train_malicious/mutate_delete_two/pickles --batch_size 50 --test robustness_spec/seed_test_malicious/mutate_delete_two/pickles --test_batches 674 --model_name robust_delete_two
```

#### Robust B
* Train:
```
time python train_insert.py --train robustness_spec/seed_train_malicious/mutate_insert_any_pt1/pickles --test robustness_spec/seed_test_malicious/mutate_insert_rootany/pickles --test_batches 5728 --model_name robust_insert_one
```
```
time python train_insert.py --resume --train robustness_spec/seed_train_malicious/mutate_insert_any_pt2/pickles --test robustness_spec/seed_test_malicious/mutate_insert_rootany/pickles --test_batches 2869 --model_name robust_insert_one
```

#### Robust C
* Train:
```
time python train_delete.py --train robustness_spec/seed_train_malicious/mutate_delete_two/pickles --batch_size 50 --test robustness_spec/seed_test_malicious/mutate_delete_two/pickles --test_batches 674 --model_name robust_delete_two
```

#### Robust D
* Train:
```
time python train_insert.py --train robustness_spec/seed_train_malicious/mutate_insert_rootallbutone_pt1/pickles --test robustness_spec/seed_test_malicious/mutate_insert_rootallbutone/pickles --test_batches 2869 --model_name robust_insert_allbutone >! ../models/adv_trained/robust_insert_allbutone.log 2>&1&
```
```
time python train_insert.py --resume --train robustness_spec/seed_train_malicious/mutate_insert_rootallbutone_pt2/pickles --test robustness_spec/seed_test_malicious/mutate_insert_rootallbutone/pickles --test_batches 2869 --model_name robust_insert_allbutone >> ../models/adv_trained/robust_insert_allbutone.log; python train_insert.py --resume --train robustness_spec/seed_train_malicious/mutate_insert_rootallbutone_pt3/pickles --test robustness_spec/seed_test_malicious/mutate_insert_rootallbutone/pickles --test_batches 2869 --model_name robust_insert_allbutone >> ../models/adv_trained/robust_insert_allbutone.log; python train_insert.py --resume --train robustness_spec/seed_train_malicious/mutate_insert_rootallbutone_pt4/pickles --test robustness_spec/seed_test_malicious/mutate_insert_rootallbutone/pickles --test_batches 2869 --model_name robust_insert_allbutone >> ../models/adv_trained/robust_insert_allbutone.log;
```

#### Robust E
* Train:
```
time python train_insert_monotonic.py --model_name robust_monotonic
```

#### Robust A+B
* Train:
```
time python train_insert_monotonic.py --train robustness_spec/seed_train_malicious/mutate_insert_any_pt1/pickles --model_name robust_combine_two > ../models/adv_trained/robust_combine_two.log 2>&1&
```
```
time python train_insert_monotonic.py --resume --train robustness_spec/seed_train_malicious/mutate_insert_any_pt2/pickles --model_name robust_combine_two >> ../models/adv_trained/robust_combine_two.log 2>&1&
```
Then pick the best one from the last few epochs.

#### Robust A+B+E
* Train:
```
time python train_combine_monotonic.py --train robustness_spec/seed_train_malicious/mutate_insert_any_pt1/pickles --model_name robust_combine_three >! ../models/adv_trained/robust_combine_three.log 2>&1&
```
```
time python train_combine_monotonic.py --resume --train robustness_spec/seed_train_malicious/mutate_insert_any_pt2/pickles --model_name robust_combine_three >> ../models/adv_trained/robust_combine_three.log 2>&1&
```
Then pick the best one from the last few epochs.


#### Evaluate the five properties
Substitue baseline_adv_delete_one with any model name.
Sequence: property A, property B, property C, property D, property E.
```
model=$(echo baseline_adv_delete_one); python train_delete.py --evaluate --model_name $model --test robustness_spec/seed_test_malicious/mutate_delete_one/pickles --test_batches 313; python train_insert.py --evaluate --model_name $model --test robustness_spec/seed_test_malicious/mutate_insert_rootany/pickles --test_batches 2869; python train_delete.py --evaluate --model_name $model --test robustness_spec/seed_test_malicious/mutate_delete_two/pickles --test_batches 674; python train_insert.py --evaluate --model_name $model --test robustness_spec/seed_test_malicious/mutate_insert_rootallbutone/pickles --test_batches 2869; python train_insert_monotonic.py --evaluate --model_name $model
```

## Ensemble models

### Ensemble A+B
* Train the base classifier:
```
python train_adv_twocls.py --train_ben robustness_spec/seed_train_benign/mutate_delete_one/pickles/ --test_ben robustness_spec/seed_test_benign/mutate_delete_one/pickles/ --train_mal robustness_spec/seed_train_malicious/mutate_delete_one/pickles/ --test_mal robustness_spec/seed_test_malicious/mutate_delete_one/pickles/ --model_name adv_del_twocls
```
* Evaluate the properties:
Construct the ensemble and test Property B:
```
python classifier_ensemble.py --model_name adv_del_twocls
```
Test Property A, C, and E:
```
python eval_ensemble_insone.py --model_name adv_del_twocls
```
Test Property D:
```
python eval_ensemble_insone_propd.py --model_name adv_del_twocls
```

### Ensemble D
* Train the base classifier:
```
python train_adv_twocls.py --train_ben robustness_spec/seed_train_benign/mutate_keep_one/pickles/ --test_ben robustness_spec/seed_test_benign/mutate_keep_one/pickles/ --train_mal robustness_spec/seed_train_malicious/mutate_keep_one/pickles/ --test_mal robustness_spec/seed_test_malicious/mutate_keep_one/pickles/ --model_name adv_keep_twocls
```
* Construct the ensemble:
```
python classifier_ensemble_d.py --model_name adv_keep_twocls
```
* Evaluate the properties:
```
python eval_ensemble_keepone.py --model_name adv_keep_twocls
```

## XGBoost monotonic models
* Train:
  * 10 Trees `python monotonic.py --num_trees 10 --model_name model_10learner_test`
  * 100 Trees `python monotonic.py --num_trees 100 --model_name model_100learner`
  * 1K Trees `python monotonic.py --num_trees 1000 --model_name model_1000learner`
  * 2K Trees `python monotonic.py --num_trees 2000 --model_name model_2000learner`
* Evaluate the properties:
  * Test Property A: `python monotonic_test_deletion_one.py --model $model_name`
  * Test Property C: `python monotonic_test_deletion_two.py --model $model_name`

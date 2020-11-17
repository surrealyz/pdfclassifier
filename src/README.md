### Training Scripts

The mini-batch numbers are calculated to be 20 epochs below.

#### Interval Datasets
Please download the interval datasets from [this link](). Extract it under the `robustness_spec/` directory.

#### Baseline NN

* Train `python train.py --baseline --batches 5276`
* Evaluate Property A:
```
python train_delete.py --baseline --evaluate --test robustness_spec/seed_test_malicious/mutate_delete_one/pickles --test_batches 315 --seed_feat robustness_spec/seed_test_malicious/feat_dict.pickle --exploit_spec ../data/traintest_all_500test/exploit_spec/test_malicious.pickle
```

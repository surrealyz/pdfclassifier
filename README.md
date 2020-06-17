# On Training Robust PDF Malware Classifiers

Code for our paper [On Training Robust PDF Malware Classifiers](https://arxiv.org/abs/1904.03542) (Usenix Security'20)
Yizheng Chen, Shiqi Wang, Dongdong She, Suman Jana

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

The extracted training and testing libsvm files are [here](https://github.com/surrealyz/pdfclassifier/tree/master/data).

[500 seed malware hash list.](https://github.com/surrealyz/pdfclassifier/blob/master/data/seeds_hash_list.txt)

## Robust models

## Training code

## Baseline comparison

## Attacks

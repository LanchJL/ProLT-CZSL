# Revealing the Proximate Long-Tail Distribution in Compositional Zero-Shot Learning

Codes of Revealing the Proximate Long-Tail Distribution in Compositional Zero-Shot Learning (AAAI2024)

## Prepare
```shell
$ cd repository
$ pip install -r requirements.txt
```
## Datasets
The splits of dataset and its attributes can be found in utils/download_data.sh, the complete installation process can be found in [CGE&CompCos](https://github.com/ExplainableML/czsl).
You can download the datasets using
```shell
bash utils/download_data.sh
```
And you can set the --DATA_FOLDER in flag.py as your dataset path.

## Training
For training our model, you can run
```shell
python train.py --config [config].yml
```

We will released the checkpoint of our trained model as soon as possible.
## Test
If you want to test based on an existing weights file, please set the corresponding path in test.py to the weight file obtained from training, then run
```shell
python test.py --config [config].yml
```
## Acknowledgements
Our overall code is built on top of [CGE&CompCos](https://github.com/ExplainableML/czsl), and we sincerely appreciate the great help this work has given us.

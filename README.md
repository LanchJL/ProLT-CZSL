# Revealing the Proximate Long-Tail Distribution in Compositional Zero-Shot Learning

Codes of Revealing the Proximate Long-Tail Distribution in Compositional Zero-Shot Learning

## Prepare
```shell
$ cd repository
$ pip install -r requirements.txt
```
Datasets can be download via utils/download_data.sh according to [1].

Or it can be found in CGE[1] https://github.com/ExplainableML/czsl.

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



[1] Naeem M F, Xian Y, Tombari F, et al. Learning graph embeddings for compositional zero-shot learning[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 953-962.
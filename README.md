# MELU-Pytorch
Unofficial PyTorch implementation of MELU from the paper:
[MeLU: Meta-Learned User Preference Estimator for Cold-Start Recommendation](https://arxiv.org/abs/1908.00413v1).


## Platform
- python: 3.5+
- Pytorch: 1.0+

## Model
We offer the training process and model in multi_result_files/9b8290dd3f63cbafcd141ba21282c783.pkl.

## How to run

### Training
```python
python3 maml.py
```
You can modify the detailed parameters according to the definition in maml.py.

### Testing
```python
python3 maml.py --test
```
By default, you can directly add the test argument to test the model obtained from the same aruguments setting.
```python
mode_path = utils.get_path_from_args(args)
```
You can also modify the code in maml.py manually since the arguments may vary for training and testing process.
```python
mode_path = '9b8290dd3f63cbafcd141ba21282c783'
```

## Benchmark
The official code doesn't offer evaluation code for testing. So based on this implementation, you can test the MAE for 4 partitions. In addition, we find the hyperparameters setting in original paepr isn't reasonable so we modify that and rerun the test.

| Partition                     |  MAE based on our hyperparameters   |Reported MAE in original paper|
|---------------------------|--------------------------|--------------------------|
| Existing items for existing users |0.68$\pm$0.01|0.75$\pm $NA|
| Existing items for new users       |0.74$\pm$0.01| 0.79$\pm$NA|
| new items for existing users       |0.90$\pm$0.01| 0.92$\pm$NA|
| new items for new users             |0.89$\pm$0.02| 0.92$\pm$NA|


# Acknowledgement.
This code refers code from:
[wyharveychen/CloserLookFewShot](https://github.com/wyharveychen/CloserLookFewShot).
[lmzintgraf/cavia](https://github.com/lmzintgraf/cavia).
[hoyeoplee/MeLU](https://github.com/hoyeoplee/MeLU).


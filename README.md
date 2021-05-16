# Image Classification

## Requirements

```
numpy
torch==1.8.0
torchvision==0.9.0
Pillow
sklearn
```



## Process

This repo is used for image classification.  `train_cls.py` is used for model training. `test_cls.py` is used for testing. 

Before Training, you should prepare the training data as follows:

```
-DATA
	-cat
		-1.jpg
		-2.jpg
		-3.jpg
		......
	-dog
		-1.jpg
		-2.jpg
		-3.jpg
		......
```

Then you set the `pic_dir`  in the `train_cls.py` and `test_cls.py`.

More details can find in the code comments.
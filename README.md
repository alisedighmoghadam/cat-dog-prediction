<!-- Python -->

# Cat vs. Dog Image Classifier

A convolutional neural network built from scratch with TensorFlow/Keras that classifies photos as **cat** or **dog**, plus a small helper that shows an image and prints the model's prediction.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)

## Model

```
Input 150×150×3
→ Conv 32 → MaxPool
→ Conv 64 → MaxPool
→ Conv 128 → MaxPool
→ Flatten → Dense 512 → Dropout 0.5
→ Dense 1 (sigmoid)
```

- Loss: binary cross-entropy · Optimiser: Adam · 20 epochs · batch size 32
- Images rescaled to `[0, 1]` with `ImageDataGenerator`
- Trained on 2,000 labelled images; the model is saved as `cat_dog_cnn.h5`

## Results

| | Accuracy |
| --- | --- |
| Training (epoch 20) | 99.6% |
| Hold-out set (10 images) | 70% |
| Sample predictions in the notebook | 5 / 6 correct |

The gap between training and hold-out accuracy shows the network overfits a dataset of this size.

## Next steps

- [ ] Data augmentation (flips, rotations, zoom)
- [ ] A larger validation set and early stopping
- [ ] Transfer learning from a pretrained backbone such as MobileNetV2

## Run it

Organise images like this (the image folders are not included in the repo):

```
train/
  cats/
  dogs/
test/
  cats/
  dogs/
```

```bash
git clone https://github.com/alisedighmoghadam/cat-dog-prediction.git
cd cat-dog-prediction
pip install tensorflow opencv-python matplotlib numpy jupyter
jupyter notebook cat-dog-prediction.ipynb
```

Predict a single image:

```python
predict_cat_or_dog("my_photo.jpg")   # shows the image and prints "Cat" or "Dog"
```

Built as part of DPS920 coursework at Seneca Polytechnic.

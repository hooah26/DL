from PIL import Image
import glob
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('../../models/cat_and_dog_binary_classfication0.86.h5')
model.summary()
img_dir = '../../datasets/train/'
image_w = 64
image_h = 64
categories = ['cat', 'dog']

dog_files = glob.glob(img_dir + 'dog*')
dog_sample = np.random.randint((len(dog_files)))
dog_sample_path = dog_files[dog_sample]

cat_files = glob.glob(img_dir + 'cat*')
cat_sample = np.random.randint((len(cat_files)))
cat_sample_path = cat_files[cat_sample]

print(dog_sample_path)

try:
    img = Image.open(dog_sample_path)
    img.show()
    img = img.convert('RGB')
    img = img.resize((image_w, image_h))
    data = np.asarray(img) # type을 바꿔준다, array -> 새로 만든다
    data = data / 255
    dog_data = data.reshape(1, 64, 64, 3)

    img = Image.open(cat_sample_path)
    img.show()
    img = img.convert('RGB')
    img = img.resize((image_w, image_h))
    data = np.asarray(img) # type을 바꿔준다, array -> 새로 만든다
    data = data / 255
    cat_data = data.reshape(1, 64, 64, 3)
except:
    print('error')
print('Dog Data :',categories[int(model.predict(dog_data).round()[0][0])])

print('Cat Data :',categories[int(model.predict(cat_data).round()[0][0])])
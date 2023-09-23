import pandas as pd
import tensorflow as tf
import imgaug.augmenters as iaa
from tensorflow.data import AUTOTUNE
from typing import List

augment_config = [
            iaa.Sometimes(0.5, 
                iaa.AddToBrightness((-30, 30))),
            iaa.OneOf([
                iaa.Fliplr(1.0)
            ]),
            iaa.OneOf([
                iaa.Affine(rotate=(-20, 20)),
                iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)})
            ])
        ]

augmenter = iaa.Sequential(augment_config, random_order=True)
AUGMENTER = iaa.Sometimes(0.8, augmenter)

def load_data(path, label, target_size):
    # NOTE: normalization with /255 is done after augmentation cuz imgaug lib needs image to be uint8
    image = tf.io.read_file(path)
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.resize(image, target_size)
    label = tf.cast(label, tf.int32)
    return image, label

def normalize(images, label):
    result = tf.cast(images, tf.float32)
    result = result / 255.0
    return result, label

def augment(x, y, augmenter):
    # imgaug require uint8 as input
    x = tf.cast(x, tf.uint8)
    # augment_image for a single image
    # augment_imageS for a batch of image
    x = tf.numpy_function(augmenter.augment_image,
                           [x],
                           x.dtype)
    return x, y

def build_data_pipeline(annot_df: pd.DataFrame, classes: List[str], split: str, img_size: List[int],
                        batch_size: int=8, do_augment: bool=False, augmenter: iaa=None):
    df = annot_df[annot_df['split']==split]
    path = df['abs_path']
    label = df[classes]
    
    pipeline = tf.data.Dataset.from_tensor_slices((path, label))
    pipeline = (pipeline
                .shuffle(len(df))
                .map(lambda path, label: load_data(path, label, target_size=img_size), 
                                                          num_parallel_calls=AUTOTUNE)
               )
                            
    if do_augment and augmenter:
        pipeline = pipeline.map(lambda x,y: augment(x, y, augmenter), num_parallel_calls=AUTOTUNE)
    
    pipeline = (pipeline
                .map(normalize, num_parallel_calls=AUTOTUNE)
                .batch(batch_size)
                .prefetch(AUTOTUNE)
               )
                
    return pipeline
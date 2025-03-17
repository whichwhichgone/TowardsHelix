import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np


dst_path = '/liujinxin/code/rlds_dataset_builder/tensorflow_datasets'
builder = tfds.builder('dummy_data_ur5', data_dir=dst_path)
ds = builder.as_dataset(split='train', shuffle_files=False)

for example in ds.take(1):
    actions = []
    images = []
    for step in example['steps']:
        actions.append(step['action'].numpy())
        images.append(step['observation']['image'].numpy())

    caption = step['language_instruction'].numpy().decode()
    image_strip = np.concatenate(images[::10], axis=1)

    plt.figure(figsize=(20, 20))
    plt.imshow(image_strip)
    plt.title(caption)
    plt.savefig(f'./imgs_debug/image.png', dpi=300)
    plt.close()
    print(f"action length is {len(actions)}")

    save_action = np.array(actions)
    np.save(f"action_test.npy", save_action)

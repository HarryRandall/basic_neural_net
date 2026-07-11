from pathlib import Path

import tensorflow as tf
from PIL import Image


output_dir = Path("digits")
output_dir.mkdir(exist_ok=True)

(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

for digit in range(10):
    index = next(i for i, label in enumerate(y_test) if label == digit)
    image = Image.fromarray(x_test[index].astype("uint8"), mode="L")
    image.save(output_dir / f"digit{digit + 1}.png")

print(f"Exported 10 images to {output_dir}/")

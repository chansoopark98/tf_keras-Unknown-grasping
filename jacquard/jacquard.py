"""cornell_grasp dataset."""
import tensorflow_datasets as tfds
import os
import glob
import tensorflow as tf
import numpy as np
import io
import tifffile as tiff
from torch import from_file
# import .utils.dataset_processing.grasp as grasp
# import .utils.dataset_processing.image as image
from utils.dataset_processing import grasp, image




def _gr_text_to_no(l, offset=(0, 0)):
  """
  Transform a single point from a Cornell file line to a pair of ints.
  :param l: Line from Cornell grasp file (str)
  :param offset: Offset to apply to point positions
  :return: Point [y, x]
  """
  x, y = l.split()
  return [int(round(float(y))) - offset[0], int(round(float(x))) - offset[1]]

# root_dir = os.path.abspath(os.sep)
dir = '/home/park/park'

# TODO(cornell_grasp): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(cornell_grasp): BibTeX citation
_CITATION = """
"""
GLOBAL_SHAPE = None

class Jacquard(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for cornell_grasp dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  MANUAL_DOWNLOAD_INSTRUCTIONS = '/home/park/park/'

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(cornell_grasp): Specifies the tfds.core.DatasetInfo object
    
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'rgb': tfds.features.Image(shape=(None, None, 3)),
            'depth': tfds.features.Tensor(shape=(1024, 1024, 1), dtype=tf.float32),
            'box': tfds.features.Tensor(shape=(None, 4, 2), dtype=tf.float32),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        # supervised_keys=('input', "depth", "box"),  # Set to `None` to disable
        supervised_keys=None,
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(cornell_grasp): Downloads the data and defines the splits
    archive_path = dl_manager.manual_dir / 'jacquard.zip'
    extracted_path = dl_manager.extract(archive_path)

    # TODO(cornell_grasp): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(img_path=extracted_path/'rgb', label_path=extracted_path/'depth', box_path=extracted_path/'box')
    }

  def _generate_examples(self, img_path, label_path, box_path):
    """Yields examples."""
    img = os.path.join(img_path, '*.png')
    label = os.path.join(label_path,'*.tif')
    box = os.path.join(box_path, '*.txt')
    
    img_files = glob.glob(img)
    img_files.sort()
    label_files = tf.io.gfile.glob(label)
    label_files.sort()
    box_files = glob.glob(box)
    box_files.sort()


  
    for i in range(len(img_files)):
      gtbbs = grasp.GraspRectangles.load_from_jacquard_file(box_files[i], scale=224/1024.)
      gtbbs = gtbbs.to_array()
      bbs = np.array(gtbbs)
      bbs = tf.convert_to_tensor(gtbbs, tf.float32)
      bbs =bbs.numpy()
      yield i, {
          'rgb': img_files[i],
          'depth': self._load_tif(label_files[i]),
          'box' : bbs
      }

  def _load_tif(self, filename: str) -> np.ndarray:
    """Loads TIF file and returns as an image array in [0, 1]."""
    with tf.io.gfile.GFile(filename, "rb") as fid:
      # img = tfds.core.lazy_imports.skimage.external.tifffile.imread(
          # io.BytesIO(fid.read())).astype(np.float32)
      img = tiff.imread(io.BytesIO(fid.read())).astype(np.float32)
      img = np.expand_dims(img , axis=-1)
    # img = (img - min_per_channel) / (max_per_channel - min_per_channel) * 255
    # img = np.clip(img, 0, 255).astype(np.uint8)
    return img


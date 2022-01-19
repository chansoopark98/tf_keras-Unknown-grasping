"""cornell_grasp dataset."""
import tensorflow_datasets as tfds
import os
import glob
import tensorflow as tf

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

class CornellGrasp(tfds.core.GeneratorBasedBuilder):
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
            'depth': tfds.features.Tensor(shape=(None, None, 1), dtype=tf.float64, encoding=tfds.features.Encoding),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('rgb', 'depth'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(cornell_grasp): Downloads the data and defines the splits
    archive_path = dl_manager.manual_dir / 'data.zip'
    extracted_path = dl_manager.extract(archive_path)

    # TODO(cornell_grasp): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(img_path=extracted_path/'rgb', label_path=extracted_path/'depth')
    }

  def _generate_examples(self, img_path, label_path):
    """Yields examples."""
    img = os.path.join(img_path, '*.png')
    label = os.path.join(label_path,'*.tif')
    img_files = glob.glob(img)
    label_files = glob.glob(label)

    for i in range(len(img_files)):
        yield i, {
            'rgb': img_files[i],
            # 'depth': tf.convert_to_tensor(label_files[i], tf.float64)
            'depth': label_files[i]
        }

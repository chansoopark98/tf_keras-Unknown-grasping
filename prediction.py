import tensorflow as tf
from tensor2tensor import data_generators
import os
import urllib

def download_report_hook(count, block_size, total_size):
  """Report hook for download progress.
  Args:
    count: current block number
    block_size: block size
    total_size: total size
  """
  percent = int(count * block_size * 100 / total_size)
  print("\r%d%%" % percent + " completed", end="\r")


def maybe_download(directory, filename, uri):
  """Download filename from uri unless it's already in directory.
  Copies a remote file to local if that local file does not already exist.  If
  the local file pre-exists this function call, it does not check that the local
  file is a copy of the remote.
  Remote filenames can be filepaths, any URI readable by tensorflow.gfile, or a
  URL.
  Args:
    directory: path to the directory that will be used.
    filename: name of the file to download to (do nothing if it already exists).
    uri: URI to copy (or download) from.
  Returns:
    The path to the downloaded file.
  """
  tf.gfile.MakeDirs(directory)
  filepath = os.path.join(directory, filename)
  if tf.gfile.Exists(filepath):
    tf.logging.info("Not downloading, file already found: %s" % filepath)
    return filepath

  tf.logging.info("Downloading %s to %s" % (uri, filepath))
  try:
    tf.gfile.Copy(uri, filepath)
  except tf.errors.UnimplementedError:
    if uri.startswith("http"):
      inprogress_filepath = filepath + ".incomplete"
      inprogress_filepath, _ = urllib.urlretrieve(
          uri, inprogress_filepath, reporthook=download_report_hook)
      # Print newline to clear the carriage return from the download progress
      print()
      tf.gfile.Rename(inprogress_filepath, filepath)
    else:
      raise ValueError("Unrecognized URI: " + filepath)
  statinfo = os.stat(filepath)
  tf.logging.info("Successfully downloaded %s, %s bytes." %
                  (filename, statinfo.st_size))
  return filepath


BASE_URL = "https://storage.googleapis.com/brain-robotics-data/push/"
DATA_TRAIN = (264, "push_train/push_train.tfrecord-{:05d}-of-00264")
def get_urls(count, url_part):
    template = os.path.join(BASE_URL, url_part)
    return [template.format(i) for i in range(count)]

urls = get_urls(DATA_TRAIN[0], DATA_TRAIN[1])
for url in urls:
    path = generator_utils.maybe_download(tmp_dir, os.path.basename(url), url)
    for frame_number, frame, state, action in self.parse_frames(path):
    yield {
        "frame_number": [frame_number],
        "frame": frame,
        "state": state,
        "action": action,
    }
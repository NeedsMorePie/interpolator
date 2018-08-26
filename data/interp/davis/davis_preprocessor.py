import glob
import os.path
from data.interp.interp_data_preprocessor import InterpDataPreprocessor
from PIL import Image
from io import BytesIO


class DavisDataSetPreprocessor(InterpDataPreprocessor):
    def __init__(self, tf_record_directory, inbetween_locations, shard_size=1, validation_size=0, max_shot_len=10,
                 verbose=False):
        super().__init__(tf_record_directory, inbetween_locations, shard_size, validation_size=validation_size,
                         max_shot_len=max_shot_len, verbose=verbose)

    def process_image(self, filename):
        """
        Overriden.
        """
        # https://stackoverflow.com/questions/31826335/how-to-convert-pil-image-image-object-to-base64-string
        buffered = BytesIO()
        im = Image.open(filename)
        width, height = im.size
        im.save(buffered, format='JPEG')
        bytes = buffered.getvalue()
        buffered.close()
        return bytes, height, width

    def get_data_paths(self, raw_directory):
        """
        Overriden.
        Gets the paths of images from a directory that is organized with each video shot in its own folder.
        The image order for each sequence must be obtainable by sorting their names.
        :return: List of list of image names, where image_paths[0][0] is the first image in the first video shot.
        """
        image_names = []
        extensions = ['*.jpg', '*.png']
        for item in sorted(os.listdir(raw_directory)):
            path = os.path.join(raw_directory, item)
            if os.path.isdir(path):
                cur_names = []
                for ext in extensions:
                    cur_names += glob.glob(os.path.join(path, '**', ext), recursive=True)
                if len(cur_names) > 0:
                    cur_names.sort()
                    image_names.append(cur_names)
        return image_names

import glob
import os.path
from data.interp.interp_data import InterpDataSet
from utils.img import read_image


class DavisDataSet(InterpDataSet):

    def __init__(self, directory, inbetween_locations, batch_size=1, maximum_shot_len=10):
        """
        See InterpDataSet.
        """
        super().__init__(directory, inbetween_locations, batch_size=batch_size, maximum_shot_len=maximum_shot_len)

    def _process_image(self, filename):
        """
        Overriden.
        TODO Consider cropping or downsizing the image first as they can be quite large.
        """
        with open(filename, 'rb') as fp:
            return fp.read()

    def _get_data_paths(self, raw_directory):
        """
        Overriden.
        Gets the paths of images from a directory that is organized with each video shot in its own folder.
        The image order for each sequence must be obtainable by sorting their names.
        :return: List of list of image names, where image_paths[0][0] is the first image in the first video shot.
        """
        image_names = []
        extensions = ['*.jpg']
        for item in os.listdir(raw_directory):
            path = os.path.join(raw_directory, item)
            if os.path.isdir(path):
                cur_names = []
                for ext in extensions:
                    cur_names += glob.glob(os.path.join(path, '**', ext), recursive=True)
                if len(cur_names) > 0:
                    cur_names.sort()
                    image_names.append(cur_names)
        return image_names
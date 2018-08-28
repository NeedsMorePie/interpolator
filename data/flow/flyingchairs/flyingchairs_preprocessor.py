import glob
import os.path
from data.flow.flow_data_preprocessor import FlowDataPreprocessor


class FlyingChairsFlowDataPreprocessor(FlowDataPreprocessor):
    def __init__(self, directory, validation_size=1, max_flow=1000.0, shard_size=1, verbose=False):
        super().__init__(directory, validation_size=validation_size, max_flow=max_flow, shard_size=shard_size,
                         verbose=verbose)

    def get_data_paths(self):
        """
        Gets the paths of [image_a, image_b, flow] tuples from a typical flying chairs flow data directory structure.
        :return: List of image_path strings, list of flow_path strings.
        """
        images_a = glob.glob(os.path.join(self.directory, '**', '*_img1.ppm'), recursive=True)
        if self.verbose:
            print('Sorting file paths...')
        images_a.sort()
        images_b = [image_a.replace('img1', 'img2') for image_a in images_a]
        flows = [image_a.replace('img1', 'flow').replace('ppm', 'flo') for image_a in images_a]
        return images_a, images_b, flows

import glob
import os.path
from data.flow.flow_data_preprocessor import FlowDataPreprocessor


class FlyingThingsFlowDataPreprocessor(FlowDataPreprocessor):
    def __init__(self, directory, validation_size=1, max_flow=1000.0, shard_size=1, verbose=False):
        super().__init__(directory, validation_size=validation_size, max_flow=max_flow, shard_size=shard_size,
                         verbose=verbose)

    def get_data_paths(self):
        """
        Gets the paths of [image_a, image_b, flow] tuples from a typical flying things flow data directory structure.
        :return: List of image_path strings, list of flow_path strings.
        """
        # Get sorted lists.
        images = glob.glob(os.path.join(self.directory, '**', 'TRAIN', '**', 'left', '*.png'), recursive=True)
        flows = glob.glob(os.path.join(self.directory, '**', 'TRAIN', '**', 'into_future', 'left', '*.pfm'),
                          recursive=True)
        if self.verbose:
            print('Sorting file paths...')
        images.sort()
        flows.sort()
        assert len(images) == len(flows)
        if self.verbose:
            print('Filtering file paths...')
        # Make sure the tuples are all under the same directory.
        filtered_images_a = []
        filtered_images_b = []
        filtered_flows = []
        for i in range(len(images) - 1):
            directory_a = os.path.dirname(images[i])
            directory_b = os.path.dirname(images[i + 1])
            if directory_a == directory_b:
                filtered_images_a.append(images[i])
                filtered_images_b.append(images[i + 1])
                filtered_flows.append(flows[i])
        return filtered_images_a, filtered_images_b, filtered_flows

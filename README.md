# interpolator

## Install and run

### Prerequisites

*   Anaconda3
    *   tensorflow or tensorflow-gpu
    *   opencv-python
    *   matplotlib
    *   joblib
    
### Running tests

#### Run all tests

Run this command:

```
python run_tests.py
```

#### Run a specific test

Run this command:

```
python -m unittest path.to.<name>_test
```

For example:

```
python -m unittest pwcnet.warp.warp_test
```

### Running the PWCNet training pipeline

#### Creating a TFRecord optical flow dataset

1.  Download a dataset (i.e. Sintel or FlyingChairs).

2.  Organize the files into the following format:

    For Sintel this might look like:
    ```
    training_dataset
        clean
            alley_1
                image_0000.png
                ...
            alley_2
            ...
            temple_3
        flow
            alley_1
                flow_0000.flo
                ...
            alley_2
            ...
            temple_3
    ```
    
    For FlyingChairs this might look like:
    ```
    training_dataset
        data
            00000_img1.ppm
            00000_img2.ppm
            00000_flow.flo
            00001_img1.ppm
            00001_img2.ppm
            00001_flow.flo
            ...
    ```
    
    For FlyingThings this might look like:
    ```
    training_dataset
        frames
            TEST
            TRAIN
                <set>
                    <clip>
                        left
                            0000.png
                            ...
                        right
        optical_flow
            TEST
            TRAIN
                <set>
                    <clip>
                        into_future
                            left
                                OpticalFlowIntoFuture_0000_L.pfm
                                ...
                            right
                        into_past
    ```
    
3.  Run the following command from the project root directory:

    For Sintel:
    ```
    python -m mains.create_flow_dataset --directory="<path>/<to>/<training_dataset>" --num_validation=100 --shard_size=1 --data_source="sintel"
    ```
    
    For FlyingChairs:
    ```
    python -m mains.create_flow_dataset --directory="<path>/<to>/<training_dataset>" --num_validation=100 --shard_size=1 --data_source="flyingchairs"
    ```
    
    For FlyingThings:
    ```
    python -m mains.create_flow_dataset --directory="<path>/<to>/<training_dataset>" --num_validation=100 --shard_size=1 --data_source="flyingthings"
    ```

4.  Expected output should be:

    ```
    <training_dataset_directory>
        ...
        0_flowdataset_train.tfrecords
        ...
        n_flowdataset_train.tfrecords
        
        0_flowdataset_valid.tfrecords
        ...
        n_flowdataset_valid.tfrecords
    ```
    
#### Training a PWCNet

1.  Have your tf records prepared.

2.  If you need to modify the config, make a copy of mains/configs/train_pwcnet.json.

    For FlyingChairs make sure:
    
    ```
    "crop_width": 448,
    "crop_height": 384
    ```

3.  Run the following command:

    ```
    python -m mains.train_pwcnet --directory="<path>/<to>/<tf_records>" --checkpoint_directory="<path>/<to>/<checkpoint_output>"
    ```
    
    Or if you modified the config add:
    
    ```
    --config="<path>/<to>/<config>.json"
    ```

4.  Launch tensorboard.

    ```
    tensorboard --logdir="<path>/<to>/<checkpoint_output>"
    ```

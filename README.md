# interpolator

## Install and run

### Prerequisites

*   Anaconda3
    *   tensorflow or tensorflow-gpu
    *   opencv-python
    *   matplotlib
    *   joblib
    *   pillow
    
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
    
3.  Run the following command from the project root directory:

    For Sintel:
    ```
    python -m mains.create_flow_dataset --directory="<path>/<to>/<training_dataset>" --num_validation=100 --shard_size=25 --data_source="sintel"
    ```
    
    For FlyingChairs:
    ```
    python -m mains.create_flow_dataset --directory="<path>/<to>/<training_dataset>" --num_validation=100 --shard_size=25 --data_source="flyingchairs"
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

2.  Run the following command:

    ```
    python -m mains.train_pwcnet --directory="<path>/<to>/<tf_records>" --checkpoint_directory="<path>/<to>/<checkpoint_output>" --validate_every=10000
    ```
    
    For FlyingChairs, add the following commands:
    
    ```
    --crop_width=448 --crop_height=384
    ```

3.  Launch tensorboard.

    ```
    tensorboard --logdir="<path>/<to>/<checkpoint_output>"
    ```

### Test-running the Context-Aware Interpolation training

#### Creating a TFRecord dataset

1.  Download [DAVIS](https://davischallenge.org/davis2017/code.html).

2.  If you wish to add more images, be sure that they are organized like DAVIS.
    For example, in DAVIS/JPEGImages/(480p | 1080p), the structure is:

    ```
    <directory to images>
        <shot_0>
            <image_0>.(png | jpg)
            ...
            <image_n>.(png | jpg)
        <shot_1>
        ...
        <shot_n>
    ```
    
3.  Run the following command from the project root directory:

    ```
    python -m mains.create_davis_dataset -d path/to/DAVIS -o path/to/tfrecords_dir
    ```

4.  Expected output in the specified output directory should be:

    ```
    <tfrecords_dir>
        ...
        0_interp_dataset_train.tfrecords
        ...
        n_interp_dataset_train.tfrecords
        
        0_interp_dataset_valid.tfrecords
        ...
        n_interp_dataset_valid.tfrecords
    ```
    
#### Training

1.  Have your tf records prepared.

2.  Run the following command:

    ```
    python -m mains.train_context_interp -d path/to/tfrecords_dir -c path/to/checkpoints_dir -w path/to/pwcnet_weights.npz
    ```

3.  Launch tensorboard.

    ```
    tensorboard --logdir=path/to/checkpoints_dir
    ```
# deepfake_pytorch
*JUST FOR STUDY AND RESEARCH*

## Requirement:
```
Python == 3.6
pytorch >= 0.4.0 or 1.0
```
```shell
conda config --add channels conda-forge
conda config --add channels pytorch
conda config --set show_channel_urls true
conda env create -f env-spec.yml -p env
conda activate ./env
```

pytorch-1.0 is supported.
 You need a modern GPU and CUDA support for better performance.

## How to run
```
cd #current directory#

mkdir train

cd train
```
Put videos of A and B to train/

for example: trump.mp4 and me.mp4 where A is trump, B is myself

```
mkdir personA #(To save frames of person A)
mkdir personB #(To save frames of person B)
```

### Crop frames from videos
`python crop_from_video.py`  # Make sure change Video_Path and save_path parameter in the python file. Do it twice to crop trump video to personA directory, crop myselft video to personB directory.

```
mkdir personA_face #(To save faces of person A from personA)
mkdir personB_face #(To save faces of person B from personB)
```

### Use dlib to crop faces from frames and save to personA_face and personB_face
`python crop_face.py` # Make sure change Image_Folder and OutFace_Folder parameter in the python file. Do it twice to crop trump face to personA_face directory, crop myselft face to personB_face directory.

### Train Model
`python train.py`

### Load Model and Output video with my face and trump body
`python convert_video.py`

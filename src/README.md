# Dependencies
Used versions of libraries during development:  
- tensorflow : 2.1.0
- numpy : 1.18.2
- OpenCV : 4.2.0
- scipy : 1.4.1
- tqdm: 4.44.1
- tensorboard: 2.1.1

# How to use the code

## Preparing the environment

TODO: requirements.txt  
TODO: Dockerfile

## Training the network

To train baseline network (SDDLite + MobileNet v2) use commands:
```bash
cd src
python3 train.py --epochs 50 --model 
```

The trained model will be saved in /cache folder.
For advanced options please refer to [opts.py](./src/opts.py) file and [the code](./src/).

## Preparing quantized models

TODO (*this is still int the planning phase*):
- export type: TF, TFLite
- export modes : FP32, FP16, INT8
- quantize option (post training, finetuning with training-aware)

## Running the model:
TODO (*this is still int the planning phase*):
- run on: single image, folder with images, video file, webcam


## Running the performance check:
TODO (*this is still int the planning phase*):
- test mAP 
- test FPS
- test batch_size impact
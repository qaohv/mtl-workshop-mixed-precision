Mixed precision simple demo

## Build docker image

```docker build . -t mixed-precision```

## Run container
```docker run -it --gpus=all --rm --ipc=host mixed-precision bash```

## Run training:

1. Without mixed precision: ```python main.py --batch-size 256```    
2. With mixed precision: ```python main.py --batch-size 256 --use-mixed-precision O1```
3. With resnet50, no mixed precision and huge batch we get and `CUDA out of memory message`: ```python main.py --arch resnet50 --batch-size 128```
4. With resnet50 and mixed preicsion, same huge batch training works: ```python main.py --arch resnet50 --batch-size 128 --use-mixed-precision O1```
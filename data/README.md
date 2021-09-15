## Download Model

``` bash
!wget -c https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet18-v2-7.onnx -O resnet18.onnx
!wget -c https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet101-v2-7.onnx -O resnet101.onnx
```
## Batching Model
if you want to use batching model example, run `export_onnx.py` script:

``` bash
python export_onnx.py
```

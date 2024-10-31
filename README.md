# AdCVFinal
Final project: Depth video data-enabled predictions of pig body weight using CNN and vision transformer algorithm





### Simple CNN -- the model we used in 2023Spring_DL



```
METRICS ON ENTIRE DATASET:
--------------------------
Test RMSE:	5.00520
Test R^2 Score:	0.77779
Test MAPE:	5.09345%
--------------------------
```

<p align="center">
<img src= './SimpleCNN/SimpleCNN_training.png' width='40%' height='40%'> 

### ViT--80by80
[Vit_80.ipynb](./ViT_80by80/Vit_80.ipynb)

```
METRICS ON ENTIRE DATASET:
--------------------------
Test RMSE:	6.01439
Test R^2 Score:	0.70610
Test MAPE:	5.92023%
--------------------------
```

<p align="center">
<img src= './ViT_80by80/output.png' width='40%' height='40%'> 

### ResNet50 -- with transfer learning

[ResNet50.ipynb](./ResNet50/ResNet50.ipynb)

```
METRICS ON ENTIRE DATASET:
--------------------------
Test RMSE:	4.81642
Test R^2 Score:	0.79400
Test MAPE:	4.79075%
--------------------------
```

<p align="center">
<img src= './ResNet50/output.png' width='40%' height='40%'> 

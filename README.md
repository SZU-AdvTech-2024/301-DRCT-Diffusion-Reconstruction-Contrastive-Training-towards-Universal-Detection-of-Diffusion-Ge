# DRCT: Diffusion Reconstruction Contrastive Training towards Universal Detection of Diffusion Generated Images

## DRCT-2M Dataset
数据集下载链接[modelscope](https://modelscope.cn/datasets/BokingChen/DRCT-2M/files).

### 使用DRCT-2M数据集训练ConvB
```convnext_base_in22k
./mytrain.sh
```

### 测试
- 在DRCT-2M数据集上测试并且使用FGSM攻击DRCT
```
./test_DRCT-2M.sh
```
- 在GenImage数据集上测试DRCT
```
./test_GenImage.sh
```


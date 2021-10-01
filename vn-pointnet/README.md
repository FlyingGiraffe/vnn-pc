# VN-PointNet Classification & Segmentation

This repository is the VN-PointNet model in the paper
[Vector Neurons: A General Framework for SO(3)-Equivariant Networks](https://arxiv.org/pdf/2104.12229.pdf) with all the original training setups in (the pytorch implementation of) [PointNet](https://github.com/WangYueFt/dgcnn/). A merged version with VN-DGCNN can be found [here](https://github.com/FlyingGiraffe/vnn/).

[[Project]](https://cs.stanford.edu/~congyue/vnn/) [[Paper]](https://arxiv.org/pdf/2104.12229.pdf)

## Preparation

The code structure follows [this implementation](https://github.com/yanx27/Pointnet_Pointnet2_pytorch/) of PointNet/PointNet++. Please follow their instructions to prepare the data and install the dependencies.

[Pytorch3D](https://pytorch3d.readthedocs.io/en/latest/) is needed to generate random rotations.

## Usage

### Classification on ModelNet40
To train the network, run
```
python train_cls.py --model_mode=pointnet_equi --model=pointnet_cls --log_dir=LOG_DIR --rot=ROTATION
```
with `ROTATION` be `aligned`, `z`, or `so3`. To evaluate the trained network, run
```
python test_cls.py --model_mode=pointnet_equi --model=pointnet_cls --log_dir=LOG_DIR --rot=ROTATION
```
To test the pretrained network, set `LOG_DIR` to be `pretrained/model.cls.vn_pointnet.z.pth`.

## Citation
Please cite this paper if you want to use it in your work,

    @article{deng2021vn,
      title={Vector Neurons: a general framework for SO(3)-equivariant networks},
      author={Deng, Congyue and Litany, Or and Duan, Yueqi and Poulenard, Adrien and Tagliasacchi, Andrea and Guibas, Leonidas},
      journal={arXiv preprint arXiv:2104.12229},
      year={2021}
    } 

## License
MIT License

## Acknowledgement
The structure of this codebase is borrowed [this pytorch implementation](https://github.com/yanx27/Pointnet_Pointnet2_pytorch/) of PointNet/PointNet++.
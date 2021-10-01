# VN-DGCNN Classification & Segmentation

This repository is the VN-DGCNN model in the paper
[Vector Neurons: A General Framework for SO(3)-Equivariant Networks](https://arxiv.org/pdf/2104.12229.pdf) with all the original training setups in [DGCNN](https://github.com/WangYueFt/dgcnn/). A merged version with VN-PointNet can be found [here](https://github.com/FlyingGiraffe/vnn/).

[[Project]](https://cs.stanford.edu/~congyue/vnn/) [[Paper]](https://arxiv.org/pdf/2104.12229.pdf)

## Preparation

The code structure follows [this implementation](https://github.com/AnTao97/dgcnn.pytorch/) of DGCNN. Please follow their instructions to prepare the data and install the dependencies.

[Pytorch3D](https://pytorch3d.readthedocs.io/en/latest/) is needed to generate random rotations.

## Usage

### Classification on ModelNet40
To train the network, run
```
python main_cls.py --exp_name=dgcnn_vnn --model=eqcnn --rot=ROTATION
```
with `ROTATION` be `aligned`, `z`, or `so3`. To evaluate the trained network, run
```
python main_cls.py --exp_name=dgcnn_vnn --model=eqcnn --rot=ROTATION --eval
```
To test the pretrained network, run
```
python main_cls.py --exp_name=dgcnn_vnn --model=eqcnn --rot=ROTATION --eval --model_path pretrained/model.cls.vn_dgcnn.z.t7
```

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
The structure of this codebase is borrowed from [DGCNN](https://github.com/WangYueFt/dgcnn/) as well as [this implementation](https://github.com/AnTao97/dgcnn.pytorch/).
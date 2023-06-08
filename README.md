# TEF

## Introduction
![pipeline](imgs/poster.png)
This repo contains the code for **High-fidelity Event-Radiance Recovery via Transient Event Frequency** (CVPR 2023 [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/html/Han_High-Fidelity_Event-Radiance_Recovery_via_Transient_Event_Frequency_CVPR_2023_paper.html), [[Video]](https://www.youtube.com/watch?v=wf138eAoazE)), by [Jin Han](https://hjynwa.github.io/), Yuta Asano, [Boxin Shi](https://ci.idm.pku.edu.cn/), [Yinqiang Zheng](https://scholar.google.com/citations?user=JD-5DKcAAAAJ&hl), and [Imari Sato](https://scholar.google.com/citations?user=gtfbzYwAAAAJ).

## How to run the code
We provide data samples in the folder ```./data_samples/```. The recovered radiance values will be saved in ```./data_samples/xxx/ev_radiance_360x640_len4.npy```. 

For hyperspectral reconstruction, since there are multiple event files captured under different light sources (with different narrow-band wavelengths), it takes longer time to get the radiance values in different wavelengths.

- For hyperspectral reconstruction:
    ```shell
    python TEF.py -m hyperspectral -i data_samples/painting
    ```
    Then relight using different lighting files:
    ```shell
    python relight.py -i data_samples/painting
    ```

- For depth sensing:
    ```shell
    python TEF.py -m depth -i data_samples/depth
    ```

- For iso-depth contour reconstruction:
    ```shell
    python TEF.py -m iso-contour -i data_samples/shape
    ```


## Citation
If you find the paper is useful for your research, please cite our paper as follows:

```
@InProceedings{Han_2023_CVPR,
    author    = {Han, Jin and Asano, Yuta and Shi, Boxin and Zheng, Yinqiang and Sato, Imari},
    title     = {High-Fidelity Event-Radiance Recovery via Transient Event Frequency},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {20616-20625}
}
```
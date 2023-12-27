# Experiments for the paper "4.6-bit Quantization for Fast and Accurate Neural Network Inference on CPUs"

*by Anton Trusov, Elena Limonova, Dmitry Nikolaev, and Vladimir V. Arlazarov*

This reposetry contain the sorce code for trainng 4.6-bit quantized network in pytorch, as it was used in the experiments for the paper "4.6-bit Quantization for Fast and Accurate Neural Network Inference on CPUs".

To directly reproduce the experiments run one of the following commands

- ``python3 experiment1_cifar.py -m CNN[6-10] -p [prexix] --step_epochs 50 --lr_scale 0.5 --n_exp_rep 5``

- ``python3 experiment2_resnet18.py -p [prefix] --n_exp_rep 5``

- ``python3 experiment2_resnet34.py -p [prefix] --n_exp_rep 5``

This will save results in .json format in the ``./results`` subdirectory.

``[prefix]`` may be any string. it is used as a prefix in result file names.

For ResNet experiments you might want to provide paths to training and validation sets, using ``--imagenet_train_path``
and ``--imagenet_val_path`` options.
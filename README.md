# LAFEAT attack

## Introduction
Because we take reproducibility very seriously,
along with the submitted paper,
we include in the supplementary materials
the source code for running the LAFEAT attack
on [TRADES](https://github.com/yaodongyu/TRADES).
Note that for reproducibility,
the scripts are made to be completely deterministic,
your runs should *hopefully* produce
exactly the same results as ours.

## Requirements

* Python 3 (>= 3.6)
* PyTorch (>= 1.2.0)

## Instructions for reproducing attacks on TRADES

1. Download the original TRADES CIFAR-10
   [`model_cifar_wrn.pt`](https://drive.google.com/file/d/10sHvaXhTNZGz618QmD5gSOAjO3rMzV33/view?usp=sharing)
   model [provided by the authors](https://github.com/yaodongyu/TRADES#how-to-download-our-cnn-checkpoint-for-mnist-and-wrn-34-10-checkpoint-for-cifar10),
   and place it in the `models/` folder.

2. To train logits for intermediate features,
   run the following command:
   ```sh
   python3 train.py --max-epoch=100 --save-model=trades_new
   ```
   It will run for 100 epochs
   and save the final logits model at `models/trades_new.pt`.
   We have also included trained logits
   named `models/trades.pt` with the code,
   so you can skip this step.

3. To perform a multi-targeted attack
   on the TRADES model with trained intermediate logits, run:
   ```sh
   python3 attack.py \
       --verbose --batch-size=${your_batch_size:-2000} \
       --multi-targeted --num-iterations=1000 \
       --logits-model=models/trades_new.pt  # your trained logits
   ```
   It will run a multi-targeted LAFEAT attack
   and save the adversarial images at `attacks/lafeat.{additional_info}.pt`.

4. For testing with the original TRADES
   [evaluation script](https://github.com/yaodongyu/TRADES/blob/master/evaluate_attack_cifar10.py),
   we need to first convert the adversarial examples
   for their script with the following command:
   ```sh
   python3 convert.py --name=lafeat.{additional_info}.pt
   ```
   By default,
   it converts the `.pt` file to a `cifar10_X_adv.npy` file
   and performs additional range clipping
   to ensure correct L-inf boundaries
   under the effect of floating-point errors.
   It also generates a new `attacks/cifar10_X_adv.npy` file.
   We ran multi-targeted LAFEAT with 1000 iterations,
   and generated the adversarial examples
   with a **52.94%** accuracy for the CIFAR-10 test set,
   which places it at the top of the
   [TRADES CIFAR-10 white-box leaderboard](https://github.com/yaodongyu/TRADES#white-box-leaderboard-1).
   For convenience,
   we uploaded the file anonymously,
   and you can download it from:
    * [cifar10_X_adv.npy](https://www.dropbox.com/s/ke3pi8llau1mk5a/cifar10_X_adv.npy?dl=1).

5. Download the CIFAR-10 datasets
   for TRADES’s testing script,
   and place them in the `attacks/` folder:
    * [`cifar10_X.npy`](https://drive.google.com/file/d/1PXePa721gTvmQ46bZogqNGkW31Vu6u3J/view?usp=sharing)
    * [`cifar10_Y.npy`](https://drive.google.com/file/d/1znICoQ8Ds9MH-1yhNssDs3hgBpvx57PV/view?usp=sharing)

6. Evaluate with
   [the original TRADES script](https://github.com/yaodongyu/TRADES/blob/master/evaluate_attack_cifar10.py)
   (with minor modifications to make it work with our paths)
   using:
   ```sh
   python3 eval_trades.py
   ```
   and you should be able
   to test the accuracy of LAFEAT adversarial examples
   on the TRADES model.

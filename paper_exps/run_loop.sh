#!/bin/bash
# both stepsize reductions: x0.1 at halfway and x0.8 every 1/5 total epochs


for i in {0..19}
do # ok
    CUDA_VISIBLE_DEVICES=1 python BNN.py --data=kin8nm --trial=$i --stepsize=0.5 --T=1e-5 --loggingprefix=eval
done
for i in {0..19}
do #ok
    CUDA_VISIBLE_DEVICES=1 python BNN.py --data=boston --trial=$i --stepsize=0.5 --T=0.01 --loggingprefix=eval
done
for i in {0..19}
do
    CUDA_VISIBLE_DEVICES=1 python BNN.py --data=combined --trial=$i --stepsize=0.5 --T=0.001 --loggingprefix=eval
done
for i in {0..19}
do #ok
    CUDA_VISIBLE_DEVICES=1 python BNN.py --data=wine --trial=$i --stepsize=0.5 --T=1e-5 --loggingprefix=eval
done
for i in {0..19}
do #ok
    CUDA_VISIBLE_DEVICES=1 python BNN.py --data=concrete --trial=$i --stepsize=0.5 --T=1e-3 --loggingprefix=eval
done
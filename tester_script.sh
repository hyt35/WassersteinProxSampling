#!/bin/bash
betas=(0.01 0.1 0.5 1.)
ts=(0.01 0.1 0.5 1.)
stepsizes=(0.01 0.1 0.5 1.)


for beta in "${betas[@]}"
do
    for t in "${ts[@]}"
    do
        for stepsize in "${stepsizes[@]}"
        do
            python tester_1D.py --beta=$beta --T=$t --stepsize=$stepsize --test_fn=exp_wavy
            python tester_1D.py --beta=$beta --T=$t --stepsize=$stepsize --test_fn=exp_twopole

            python tester_2D.py --beta=$beta --T=$t --stepsize=$stepsize # double banana
        done
    done
done
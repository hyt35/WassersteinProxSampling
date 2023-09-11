#!/bin/bash
betas=(0.01 0.1 0.5 1.)

stepsizes=(0.01 0.1 0.5 1.)

for beta in "${betas[@]}"
do
    for stepsize in "${stepsizes[@]}"
    do
        python tester_1D.py --beta=$beta --stepsize=$stepsize --test_fn=exp_wavy
        python tester_1D.py --beta=$beta --stepsize=$stepsize --test_fn=exp_twopole

        python tester_2D.py --beta=$beta --stepsize=$stepsize # double banana
    done
done
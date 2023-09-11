#!/bin/bash
betas=(0.01 0.1 0.5 1.)
ts=(0.01 0.1 0.5 1.)
stepsizes=(0.01 0.1 0.5 )


for beta in "${betas[@]}"
do
    for t in "${ts[@]}"
    do
        for stepsize in "${stepsizes[@]}"
        do
            echo "${beta} ${t} ${stepsize}"
            python tester_2D_energy.py --beta=$beta --T=$t --stepsize=$stepsize # double banana
        done
    done
done

for beta in "${betas[@]}"
do
    for stepsize in "${stepsizes[@]}"
    do
        echo "${beta} ${t} ${stepsize}"
        python tester_2D_energy_SGD.py --beta=$beta --T=$t --stepsize=$stepsize # double banana
    done
done
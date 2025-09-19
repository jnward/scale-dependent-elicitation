#!/bin/bash

./sweep_eval.sh --output_dir results/temperature --seed 1234

./sweep_eval.sh --output_dir results/temperature_gpqa_2 --seed 1235 --tasks gpqa_diamond_cot_zeroshot
./sweep_eval.sh --output_dir results/temperature_gpqa_3 --seed 1236 --tasks gpqa_diamond_cot_zeroshot
./sweep_eval.sh --output_dir results/temperature_gpqa_4 --seed 1237 --tasks gpqa_diamond_cot_zeroshot

./sweep_eval.sh --output_dir results/temperature_aime_2 --seed 1235 --tasks aime24,aime25
./sweep_eval.sh --output_dir results/temperature_aime_3 --seed 1236 --tasks aime24,aime25
./sweep_eval.sh --output_dir results/temperature_aime_4 --seed 1237 --tasks aime24,aime25
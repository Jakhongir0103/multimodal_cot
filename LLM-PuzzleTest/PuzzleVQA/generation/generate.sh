#!/bin/bash

pattern_names=(
    # circle_size_number
    # color_grid
    # color_hexagon
    # color_number_hexagon
    # color_overlap_squares
    # color_size_circle
    # grid_number_color
    grid_number
    # polygon_sides_color
    # polygon_sides_number
    # rectangle_height_color
    # rectangle_height_number
    # shape_morph
    # shape_reflect
    # shape_size_grid
    # shape_size_hexagon
    # # size_cycle
    # # size_grid
    # triangle
    # venn
)

for pattern_name in "${pattern_names[@]}"; do
    python -u data_generation.py create_data \
        --pattern_name="$pattern_name" \
        --path="/lid/home/saydalie/multimodal_cot/LLM-PuzzleTest/PuzzleVQA/data/train" \
        --eval_path="/lid/home/saydalie/multimodal_cot/LLM-PuzzleTest/PuzzleVQA/data/eval" \
        --limit=10000 \
        --unique \
        > "output.txt" 2>&1
done
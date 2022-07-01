# Map-based Explanations for Optimal MAPF

By Yonathan Setiawan and Martim Brandao

## Objective

To explain why the MAPF plan obtained by CBS was equal to X, not Y (where Y is a user-provided plan).

## Citation

If you use this in your research, please cite:

> Martim Brandao, Yonathan Setiawan, "**'Why Not This MAPF Plan Instead?' Contrastive Map-based Explanations for Optimal MAPF**", *XAIP 2022*.

## Example

* Go to multi-agent/inv-shortest-path
* Run the following code:
```
python explanations_multi.py 8x8_a5_ex0.yaml -v -a
```
* `-v` and `-a` are optional and for "verbose" and "animation" flags respectively.
* Examples are located in the custom_waypoints folder.

## Benchmark

* Benchmark problems are under multi-agent/inv-shortest-path/rnd_problems_2022/
* To run the various methods on the benchmark problems, use multi-agent/inv-shortest-path/benchmark.py

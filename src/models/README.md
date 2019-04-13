# Flattened CNN models

We flatten CNN architectureas (each layer module definition) to support easy network architecture reconfiguration and generation into files. After pruning using structured parameter reguarlization, each layer has varying channel dimensions. To store layers with such irregular channel dimensions, it is rather convenient to flatten each layer structure than defining them using nested loops.

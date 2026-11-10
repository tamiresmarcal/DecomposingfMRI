"""What a brain state means, by correlating it against maps we already name.

This is interpretability, not prediction: it takes the state means stage 4b
produced and asks which known networks, gradients and parcellations they line
up with. It reads no phenotype, which is why it sits under the unsupervised
model rather than beside the selections.

`fetch_reference_maps.sh` downloads the reference maps; `data/` holds the one
parcellation that ships with the repo because it has no fetcher. README.md is
the arm's own write-up.
"""

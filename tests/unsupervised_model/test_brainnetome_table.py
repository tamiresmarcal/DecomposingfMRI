"""brainnetome_subregions.csv moved into contextualization/data/ and the
reader kept the old path: notebook 08 died with FileNotFoundError."""
from fmri_decomposition.unsupervised_model.contextualization import refmaps as rm


def test_brainnetome_table_is_found_and_whole():
    t = rm.brainnetome_table(None)
    assert len(t) == 246
    assert set(t["hemi"]) == {"L", "R"}
    assert t["bn_index"].tolist() == list(range(1, 247))

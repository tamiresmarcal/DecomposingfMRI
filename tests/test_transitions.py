"""Stage 5a's pooling invariants.

Everything here is a property that, when wrong, produces tables that LOOK fine
and cannot be pooled: a matrix whose width depends on which states a cohort
happened to visit, a dwell time computed with the wrong stride, or a state set
that silently never gets a table because nothing named it.
"""
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fmri_decomposition import transitions as T


def latents(tmp_path, cols, metadata=None, n=12):
    d = pd.DataFrame({"cohort": "c", "task": "movie", "sub": "01",
                      "window_id": np.arange(n),
                      "pca0/3": 0.0, "pca1/3": 0.0, "pca2/3": 0.0,
                      "crosses_run_boundary": False})
    for name, labels in cols.items():
        d[name] = labels
    md = {k.encode(): json.dumps(v).encode() for k, v in (metadata or {}).items()}
    p = tmp_path / "data.parquet"
    pq.write_table(pa.Table.from_pandas(d, preserve_index=False)
                     .replace_schema_metadata(md), p)
    return p


class TestNStatesFor:
    def test_recorded_k_wins_over_the_labels(self):
        # The cohort visited 3 of 8 states. K is 8 -- a property of the FIT --
        # and taking it from the labels would give this cohort a 3x3 matrix
        # while another cohort got 8x8 for the same state set.
        p = None
        assert T.n_states_for("MeanShift_pca3_8", path=p,
                              labels=np.array([0, 1, 2])) == 8

    def test_provenance_beats_a_name_that_disagrees(self, tmp_path):
        p = latents(tmp_path, {"MeanShift_pca3_8": 0},
                    metadata={"clusterers": {"MeanShift_pca3_8": {"k": 5}}})
        assert T.n_states_for("MeanShift_pca3_8", path=p) == 5

    def test_falls_back_to_the_labels_only_with_a_warning(self, capsys):
        got = T.n_states_for("Odd_pca3_x", path=None,
                             labels=np.array([0, 1, 2, 3]))
        assert got == 4
        assert "WARNING" in capsys.readouterr().out

    def test_refuses_to_guess_with_nothing_to_go_on(self):
        with pytest.raises(SystemExit, match="cannot determine"):
            T.n_states_for("Odd_pca3_x", path=None, labels=None)


class TestTimeAxis:
    def test_prefers_the_recorded_stride(self, tmp_path):
        p = latents(tmp_path, {"HMM_pca3_8": 0},
                    metadata={"stride_s": 2.47, "indep_factor": 1})
        assert T.time_axis(p, "-1", n_overlaps=5) == (2.47, 1)

    def test_derives_it_for_a_sliding_window(self, tmp_path):
        p = latents(tmp_path, {"HMM_pca3_8": 0})
        assert T.time_axis(p, "60", n_overlaps=5) == (12.0, 5)

    def test_refuses_a_negative_window_with_no_recorded_stride(self, tmp_path):
        # window_s / n_overlaps would be -0.2 s, and every dwell time and switch
        # rate derived from it would come out negative without complaint.
        p = latents(tmp_path, {"HMM_pca3_8": 0})
        with pytest.raises(SystemExit, match="no `stride_s`"):
            T.time_axis(p, "-1", n_overlaps=5)


class TestDiscoverStateColumns:
    def test_finds_every_family_without_being_told(self, tmp_path):
        p = latents(tmp_path, {"ThresholdCluster_pca3_8": 0,
                               "HMM_umap3_27": 0, "MeanShift_pca3_7": 0})
        assert T.discover_state_columns(p) == [
            "HMM_umap3_27", "MeanShift_pca3_7", "ThresholdCluster_pca3_8"]

    def test_does_not_mistake_the_embeddings_for_state_columns(self, tmp_path):
        p = latents(tmp_path, {"HMM_pca3_8": 0})
        found = T.discover_state_columns(p)
        assert found == ["HMM_pca3_8"]
        assert not any("pca0" in c for c in found)


class TestShorten:
    def test_never_merges_two_names(self):
        s = pd.Series(["ThresholdCluster_pca3_8", "MeanShift_umap3_8"])
        out = T._shorten(s)
        assert out.nunique() == 2

    def test_drops_a_prefix_shared_by_all(self):
        s = pd.Series(["HMM_pca3_8", "HMM_pca3_27"])
        assert T._shorten(s).to_list() == ["8", "27"]


class TestSubjectTransitions:
    def frame(self, labels, wid=None, boundary=None):
        n = len(labels)
        return pd.DataFrame({
            "window_id": np.arange(n) if wid is None else wid,
            "s": labels,
            "crosses_run_boundary": (np.zeros(n, bool) if boundary is None
                                     else boundary)})

    def test_cells_are_p_joint_and_sum_to_one(self):
        row = T.subject_transitions(self.frame([0, 1, 0, 1, 0]), "s", 2, 1.0, 1)
        cells = [row[c] for c in T.cell_names(2)]
        assert np.isclose(sum(cells), 1.0)
        assert row["n_transitions"] == 4

    def test_every_cell_is_present_even_when_never_visited(self):
        row = T.subject_transitions(self.frame([0] * 6), "s", 3, 1.0, 1)
        assert all(c in row for c in T.cell_names(3))
        assert row["0->0"] == 1.0 and row["2->1"] == 0.0

    def test_no_transition_is_counted_across_a_gap(self):
        # window_ids 0,1,2 then 10,11: four pairs, not five.
        row = T.subject_transitions(
            self.frame([0, 0, 0, 1, 1], wid=np.array([0, 1, 2, 10, 11])),
            "s", 2, 1.0, 1)
        assert row["n_transitions"] == 3

    def test_dwell_is_measured_within_segments_not_across_them(self):
        # One run of 3 and one of 2, never a run of 5.
        row = T.subject_transitions(
            self.frame([0, 0, 0, 0, 0], wid=np.array([0, 1, 2, 10, 11])),
            "s", 2, 1.0, 1)
        assert np.isclose(row["mean_dwell_s"], 2.5)

    def test_a_subject_whose_every_frame_is_a_boundary_still_appears(self):
        row = T.subject_transitions(
            self.frame([0, 1], boundary=np.ones(2, bool)), "s", 2, 1.0, 1)
        assert set(T.cell_names(2)) <= set(row)
        assert np.isnan(row["n_transitions"]) or row["n_transitions"] == 0

    def test_dwell_scales_with_the_stride(self):
        a = T.subject_transitions(self.frame([0, 0, 1, 1]), "s", 2, 1.0, 1)
        b = T.subject_transitions(self.frame([0, 0, 1, 1]), "s", 2, 2.47, 1)
        assert np.isclose(b["mean_dwell_s"], a["mean_dwell_s"] * 2.47)

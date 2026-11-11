import torch

from src.entailment_network.verifier_common import (LABELS, VerdictModel,
                                                    decide, macro_f1,
                                                    parse_label, row_scores,
                                                    tune_thresholds)


class TestParseLabel:
    def test_aliases(self):
        assert parse_label("supported") == 0
        assert parse_label("partially_supported") == 1
        assert parse_label("not supported") == 2
        assert parse_label("refuted") == 2
        assert parse_label(1) == 1


class TestMacroF1:
    def test_perfect_score(self):
        pred = torch.tensor([0, 1, 2, 0])
        y = torch.tensor([0, 1, 2, 0])
        f1, per = macro_f1(pred, y)
        assert f1 == 1.0
        assert set(per) == set(LABELS)

    def test_all_wrong_zero(self):
        f1, _ = macro_f1(torch.zeros(4, dtype=torch.long), torch.ones(4, dtype=torch.long))
        assert f1 == 0.0


class TestVerdictModel:
    def test_output_shape_and_softmax(self):
        m = VerdictModel(d=32, n_layers=1, K=3, r=8)
        hq = torch.randn(4, 1, 32)
        he = torch.randn(4, 5, 1, 32)
        out = m(hq, he)
        assert out.shape == (4, 5, 3)
        assert torch.isfinite(out).all()

    def test_uses_last_cached_layer(self):
        m = VerdictModel(d=16, n_layers=2, K=3, r=4)
        hq1 = torch.randn(2, 2, 16)
        he1 = torch.randn(2, 3, 2, 16)
        hq2 = hq1.clone(); he2 = he1.clone()
        hq2[:, 0, :] = 999.0; he2[:, :, 0, :] = 999.0   # corrupt first cached layer
        assert torch.allclose(m(hq1, he1), m(hq2, he2))


class TestRowAggregation:
    def test_row_scores_max_pool(self):
        probs = torch.tensor([[0.9, 0.05, 0.05], [0.1, 0.1, 0.8]])
        chunk_row = torch.tensor([0, 0])
        # two chunks for row0: max per class -> [0.9,0.1,0.8]; row1 has none
        s = row_scores(probs, chunk_row, 2)
        assert s[0, 0] == 0.9 and s[0, 2] == 0.8 and s[0, 1] == 0.1
        assert s[1].sum() == 0.0

    def test_tune_thresholds_returns_valid(self):
        s = torch.rand(20, 3)
        y = torch.randint(0, 3, (20,))
        f1, ts, tp = tune_thresholds(s, y)
        assert 0 <= ts <= 1 and 0 <= tp <= 1 and 0 <= f1 <= 1

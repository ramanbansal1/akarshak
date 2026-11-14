"""Ticket 08: frontier export from resplit CSV."""

import csv

from experiments.resplit_runner import write_csv
import experiments.export_frontier as ef


def test_export_frontier_csvs(tmp_path, monkeypatch, capsys):
    src = tmp_path / "resplit.csv"
    write_csv([dict(mode="real", n_cal=100, alpha=0.1, grid=50,
                    bound="bernstein", sel_mode="macro",
                    selector="bernstein_macro", cert_rate=1.0,
                    mean_cal_coverage=0.5, mean_test_coverage=0.4,
                    test_coverage_std=0.05, violation_rate=0.0,
                    mean_test_error=0.08, n_resplits=10),
                   dict(mode="real", n_cal=200, alpha=0.2, grid=50,
                    bound="bernstein", sel_mode="macro",
                    selector="bernstein_macro", cert_rate=0.5,
                    mean_cal_coverage=0.5, mean_test_coverage=0.3,
                    test_coverage_std=0.05, violation_rate=0.1,
                    mean_test_error=0.15, n_resplits=10)], src)
    out = tmp_path / "frontier"
    monkeypatch.setattr("sys.argv",
                        ["x", str(src), "--out", str(out)])
    ef.main()
    rows = list(csv.DictReader((out / "frontier.csv").open()))
    assert len(rows) == 2
    assert rows[0]["selector"] == "bernstein_macro"
    assert (out / "ncal.csv").exists()

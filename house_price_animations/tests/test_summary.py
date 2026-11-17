import csv
from dataclasses import replace

import pytest

from house_anim.inference import inference_frames
from house_anim.models import get_spec
from house_anim.runner import ModelResult
from house_anim.summary import format_table, merge_summary_csv, result_row
from house_anim.trainers import train_silently


@pytest.fixture(scope="module")
def results(splits):
    spec = replace(get_spec("decision_tree"), steps=(1, 2, 3))
    model, final_training = train_silently(spec, splits)
    final_inference = tuple(inference_frames(model, splits, batch_size=50))[-1]
    full = ModelResult(spec.key, spec.title, final_training, final_inference, ())
    training_only = ModelResult("lasso", "Lasso", final_training, None, ())
    return full, training_only


def read_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


CONTEXT = {"rows_used": 500, "batch_size": 50}


def test_training_only_rows_leave_test_columns_blank(results):
    row = result_row(results[1], **CONTEXT)
    assert row["test_rmse_k"] == "" and row["val_rmse_k"] != ""
    assert row["rows_used"] == "500" and row["batch_size"] == "50"
    assert "—" in format_table([row])


def test_fit_columns_separate_final_model_from_sweep_total(results):
    row = result_row(results[0], **CONTEXT)
    assert float(row["fit_seconds"]) <= float(row["total_fit_seconds"])


def test_merge_upserts_rows_in_registry_order(results, tmp_path):
    path = tmp_path / "summary.csv"
    full, training_only = results
    merge_summary_csv(path, [result_row(full, **CONTEXT)])
    merge_summary_csv(path, [result_row(training_only, **CONTEXT)])
    merge_summary_csv(path, [{**result_row(full, **CONTEXT), "fit_seconds": "9.000"}])

    rows = read_rows(path)
    assert [r["model"] for r in rows] == ["lasso", "decision_tree"]
    assert rows[1]["fit_seconds"] == "9.000"
    assert not list(tmp_path.glob("*.tmp"))


def test_training_only_rerun_keeps_earlier_test_scores(results, tmp_path):
    path = tmp_path / "summary.csv"
    full = results[0]
    merge_summary_csv(path, [result_row(full, **CONTEXT)])
    training_only = ModelResult(full.key, full.title, full.final_training, None, ())
    merge_summary_csv(path, [result_row(training_only, **CONTEXT)])
    assert read_rows(path)[0]["test_rmse_k"] != ""


def test_rerun_on_different_data_replaces_the_whole_row(results, tmp_path):
    path = tmp_path / "summary.csv"
    full = results[0]
    merge_summary_csv(path, [result_row(full, **CONTEXT)])
    training_only = ModelResult(full.key, full.title, full.final_training, None, ())
    merge_summary_csv(path, [result_row(training_only, rows_used=100, batch_size=50)])
    row = read_rows(path)[0]
    assert row["rows_used"] == "100" and row["test_rmse_k"] == ""


def test_merge_skips_malformed_existing_rows(results, tmp_path):
    path = tmp_path / "summary.csv"
    path.write_text("name,score\nfoo,1\n")
    merge_summary_csv(path, [result_row(results[0], **CONTEXT)])
    assert [r["model"] for r in read_rows(path)] == ["decision_tree"]

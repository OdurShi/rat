from rat.reported import BACKBONES, TABLES, best_baseline, check_consistency, load, value


def test_tables_load_with_expected_shapes():
    sizes = {name: len(load(name)) for name in TABLES}
    assert sizes == {"T1": 36, "T4": 8, "T5": 3, "T6": 3, "S1": 11, "S3": 36, "S4": 36, "S5": 3}
    assert {row["backbone"] for row in load("T1")} == set(BACKBONES)


def test_headline_numbers():
    t1 = load("T1")
    assert value(t1, "pair_mean", backbone="Qwen2.5-VL-3B", method="rat") == 82.61
    assert best_baseline(t1, "Qwen2.5-VL-3B") == ("xmas", 79.50)
    assert best_baseline(t1, "LLaVA-1.5-7B") == ("dataprophet", 60.69)


def test_reported_numbers_are_mutually_consistent():
    assert check_consistency() == []

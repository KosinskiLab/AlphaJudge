import sys

import pytest

from alphajudge import cli


def test_aggregate_requires_summary_before_scoring(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(cli, "process_many", lambda *a, **kw: calls.append(kw))
    monkeypatch.setattr(sys, "argv", ["alphajudge", "run", "--aggregate_report", "out.pdf"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "--aggregate_report requires --summary" in capsys.readouterr().err
    assert calls == []


def test_cache_and_structure_options_reach_scoring(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "process_many", lambda *a, **kw: calls.append(kw))
    monkeypatch.setattr(sys, "argv", ["alphajudge", "run", "--no-report",
                                   "--cache_validation", "content", "--af2_structure", "unrelaxed"])
    cli.main()
    assert calls[0]["cache_validation"] == "content"
    assert calls[0]["af2_structure"] == "unrelaxed"

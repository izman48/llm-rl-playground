from playground.sandbox import run_candidate


def test_runs_and_returns_outputs():
    r = run_candidate("def f(x):\n    return x * 2\n", "f", [[2], [3]], timeout=5)
    assert r.status == "ok"
    assert [o["value"] for o in r.outputs] == [4, 6]


def test_timeout_kills_infinite_loop():
    r = run_candidate("def f(x):\n    while True:\n        pass\n", "f", [[1]], timeout=2)
    assert r.status == "timeout"


def test_missing_entry_point_reported():
    r = run_candidate("def g(x):\n    return x\n", "f", [[1]])
    assert r.status == "ok"
    assert r.outputs[0]["error"] == "missing_entry_point"


def test_expected_outputs_never_reach_sandbox():
    # A candidate that reads every file it can still cannot find the answers,
    # because the sandbox only ever receives arguments, never expected outputs.
    code = (
        "import glob\n"
        "def f(x):\n"
        "    blob = ''.join(open(p).read() for p in glob.glob('*'))\n"
        "    return blob  # whatever it can scrape; never the answer key\n"
    )
    r = run_candidate(code, "f", [[1]], timeout=5)
    assert r.status == "ok"
    # The scraped blob is just the runner source; it contains no expected values.
    assert "expected" not in str(r.outputs[0].get("value", "")).lower()

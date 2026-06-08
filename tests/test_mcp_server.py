"""Test the MCP tool logic (the *_impl functions) without needing an MCP runtime."""
from playground import tasks
from playground.mcp import server


def test_list_tasks():
    out = server.list_tasks_impl()
    assert any(t["id"] == "reverse_string" for t in out)


def test_get_task_exposes_public_only():
    g = server.get_task_impl("reverse_string")
    assert "public_cases" in g
    # Held-out tests must never be exposed through the connector.
    assert "held_out" not in g and "held_out_cases" not in g


def test_submit_reference_scores_full():
    code = tasks.get_task("reverse_string").reference
    res = server.submit_solution_impl("reverse_string", code)
    assert res["reward"] == 1.0


def test_run_qa_reports_all_safe():
    out = server.run_qa_impl()
    for row in out["results"]:
        if row["exploit"] == "honest_reference":
            assert row["reward"] == 1.0
        else:
            assert row["reward"] <= 0.20

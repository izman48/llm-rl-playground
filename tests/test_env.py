from playground import tasks
from playground.env import CodeEnv


def test_reset_returns_prompt_and_info():
    env = CodeEnv()
    obs, info = env.reset(options={"task_id": "reverse_string"})
    assert "reverse_string" in obs
    assert info["task_id"] == "reverse_string"


def test_step_with_reference_solves():
    env = CodeEnv()
    env.reset(options={"task_id": "reverse_string"})
    code = tasks.get_task("reverse_string").reference
    obs, reward, terminated, truncated, info = env.step(code)
    assert reward == 1.0
    assert terminated and not truncated


def test_agentic_mode_gives_non_leaking_feedback():
    env = CodeEnv(max_attempts=2)
    env.reset(options={"task_id": "reverse_string"})
    obs, reward, terminated, truncated, info = env.step("def reverse_string(s):\n    return s\n")
    assert not terminated  # more attempts allowed
    assert "held-out tests" in obs
    assert "expected" not in obs.lower()  # feedback must not leak answers

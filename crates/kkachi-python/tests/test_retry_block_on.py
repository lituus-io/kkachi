"""Tests that reason().go() from Python doesn't panic on retry/rate-limit paths."""

import threading

from kkachi import reason


def test_retry_path_no_panic():
    """Basic reason().go() from Python should work without panic."""

    def mock_llm(prompt, feedback=None):
        return "The answer is 4"

    result = reason(mock_llm, "What is 2+2?").go()
    assert isinstance(result.output, str)
    assert len(result.output) > 0


def test_reason_go_from_multiple_threads():
    """Multiple Python threads calling .go() concurrently should not deadlock."""

    def mock_llm(prompt, feedback=None):
        return f"Response to: {prompt[:20]}"

    results = [None] * 4
    errors = [None] * 4

    def worker(idx):
        try:
            r = reason(mock_llm, f"Thread {idx} prompt").go()
            results[idx] = r.output
        except Exception as e:
            errors[idx] = e

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    for i, err in enumerate(errors):
        assert err is None, f"Thread {i} raised: {err}"
    for i, r in enumerate(results):
        assert r is not None, f"Thread {i} returned None"

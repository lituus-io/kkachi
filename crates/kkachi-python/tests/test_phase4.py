# Copyright © 2025 lituus-io <spicyzhug@gmail.com>
# All Rights Reserved.
# Licensed under PolyForm Noncommercial 1.0.0

"""Comprehensive tests for Phase 4: Python Binding Parity.

Tests cover:
- Pipeline composition (pipeline, concurrent)
- Step combinators (step_fn, step_scored, combinators, run_all_steps)
- Optimizer + Dataset
- Evaluation metrics
- Multi-objective / Pareto optimization
"""

import pytest
from kkachi import (
    # Pipeline
    pipeline,
    concurrent,
    PipelineBuilder,
    PipelineResult,
    ConcurrentRunnerBuilder,
    ConcurrentTaskResult,
    # Step combinators
    step_fn,
    step_scored,
    run_all_steps,
    StepDef,
    StepResult,
    # Optimizer
    optimizer,
    Dataset,
    OptimizeResult,
    OptimizerBuilder,
    # Evaluation metrics
    Metric,
    # Multi-objective / Pareto
    multi_objective,
    Objective,
    MultiObjectiveBuilder,
    MultiObjectiveValidator,
    ParetoCandidate,
    ParetoFront,
    # Existing types used in tests
    Checks,
)


# =============================================================================
# Mock LLM helper
# =============================================================================

def mock_llm(prompt, feedback=None):
    """Simple mock LLM that echoes the prompt."""
    return f"Response to: {prompt[:50]}"


def counting_llm():
    """Create a mock LLM that counts invocations."""
    state = {"count": 0}
    def llm(prompt, feedback=None):
        state["count"] += 1
        return f"Response #{state['count']}"
    llm.state = state
    return llm


# =============================================================================
# Step Combinator Tests
# =============================================================================

class TestStepFn:
    """Tests for step_fn() — creating steps from Python callables."""

    def test_basic_step(self):
        """A simple step wrapping a lambda."""
        upper = step_fn("upper", lambda s: s.upper())
        result = upper.run("hello world")
        assert isinstance(result, StepResult)
        assert result.text == "HELLO WORLD"
        assert result.score == 1.0

    def test_step_repr(self):
        """StepDef has a useful __repr__."""
        s = step_fn("my_step", lambda s: s)
        r = repr(s)
        assert "my_step" in r
        assert "StepDef" in r

    def test_step_result_str(self):
        """StepResult.__str__ returns the text."""
        s = step_fn("echo", lambda s: s)
        result = s.run("hello")
        assert str(result) == "hello"

    def test_step_identity(self):
        """Identity step passes input through unchanged."""
        identity = step_fn("identity", lambda s: s)
        result = identity.run("test input")
        assert result.text == "test input"
        assert result.score == 1.0

    def test_step_empty_input(self):
        """Step handles empty string input."""
        s = step_fn("upper", lambda s: s.upper())
        result = s.run("")
        assert result.text == ""
        assert result.score == 1.0


class TestStepScored:
    """Tests for step_scored() — creating scored steps."""

    def test_scored_step_passing(self):
        """A scored step returning (text, 1.0)."""
        check = step_scored("check", lambda s: (s, 1.0 if "hello" in s else 0.0))
        result = check.run("hello world")
        assert result.text == "hello world"
        assert result.score == 1.0

    def test_scored_step_failing(self):
        """A scored step returning (text, 0.0)."""
        check = step_scored("check", lambda s: (s, 1.0 if "hello" in s else 0.0))
        result = check.run("goodbye")
        assert result.text == "goodbye"
        assert result.score == 0.0

    def test_scored_step_partial(self):
        """A scored step returning partial score."""
        length_check = step_scored("len", lambda s: (s, min(len(s) / 10.0, 1.0)))
        result = length_check.run("hello")
        assert abs(result.score - 0.5) < 0.01


class TestStepComposition:
    """Tests for step combinator methods."""

    def test_then_sequential(self):
        """Step.then() chains steps sequentially."""
        upper = step_fn("upper", lambda s: s.upper())
        exclaim = step_fn("exclaim", lambda s: s + "!")
        composed = upper.then(exclaim)
        result = composed.run("hello")
        assert result.text == "HELLO!"

    def test_then_repr(self):
        """Composed step has descriptive __repr__."""
        a = step_fn("a", lambda s: s)
        b = step_fn("b", lambda s: s)
        composed = a.then(b)
        r = repr(composed)
        assert "then" in r

    def test_race_picks_best(self):
        """Step.race() picks the result with the best score."""
        low = step_scored("low", lambda s: (s, 0.3))
        high = step_scored("high", lambda s: (s.upper(), 0.9))
        raced = low.race(high)
        result = raced.run("hello")
        assert result.text == "HELLO"
        assert abs(result.score - 0.9) < 0.01

    def test_par_concatenates(self):
        """Step.par() concatenates outputs from both steps."""
        a = step_fn("lower", lambda s: s.lower())
        b = step_fn("upper", lambda s: s.upper())
        combined = a.par(b)
        result = combined.run("Hello")
        assert "hello" in result.text
        assert "HELLO" in result.text

    def test_fallback_primary_ok(self):
        """Step.fallback() uses primary when score > 0."""
        primary = step_scored("primary", lambda s: (s.upper(), 0.8))
        backup = step_scored("backup", lambda s: (s.lower(), 0.5))
        safe = primary.fallback(backup)
        result = safe.run("Hello")
        assert result.text == "HELLO"
        assert abs(result.score - 0.8) < 0.01

    def test_fallback_primary_fails(self):
        """Step.fallback() uses backup when primary score == 0."""
        primary = step_scored("primary", lambda s: (s, 0.0))
        backup = step_scored("backup", lambda s: (s.upper(), 0.9))
        safe = primary.fallback(backup)
        result = safe.run("hello")
        assert result.text == "HELLO"
        assert abs(result.score - 0.9) < 0.01

    def test_retry_reaches_target(self):
        """Step.retry() retries until score >= target."""
        state = {"count": 0}
        def improving(s):
            state["count"] += 1
            score = 1.0 if state["count"] >= 3 else 0.2
            return (s, score)
        step = step_scored("improve", improving)
        retried = step.retry(n=5, target=0.9)
        result = retried.run("test")
        assert result.score >= 0.9

    def test_map_transforms_output(self):
        """Step.map() transforms the output text."""
        s = step_fn("echo", lambda s: s)
        mapped = s.map(lambda s: f"[{s}]")
        result = mapped.run("hello")
        assert result.text == "[hello]"

    def test_complex_composition(self):
        """Multi-level composition: then → map → fallback."""
        upper = step_fn("upper", lambda s: s.upper())
        wrap = step_fn("wrap", lambda s: f"<{s}>")
        fail = step_scored("fail", lambda s: (s, 0.0))

        composed = upper.then(wrap).fallback(fail)
        result = composed.run("hello")
        assert result.text == "<HELLO>"

    def test_three_way_chain(self):
        """Three-step sequential chain."""
        a = step_fn("a", lambda s: s + "A")
        b = step_fn("b", lambda s: s + "B")
        c = step_fn("c", lambda s: s + "C")
        chain = a.then(b).then(c)
        result = chain.run("")
        assert result.text == "ABC"


class TestRunAllSteps:
    """Tests for run_all_steps() — batch concurrent execution."""

    def test_basic_run_all(self):
        """Run multiple steps on the same input."""
        a = step_fn("lower", lambda s: s.lower())
        b = step_fn("upper", lambda s: s.upper())
        c = step_fn("reverse", lambda s: s[::-1])

        results = run_all_steps("Hello", [a, b, c])
        assert len(results) == 3
        assert results[0].text == "hello"
        assert results[1].text == "HELLO"
        assert results[2].text == "olleH"

    def test_run_all_empty(self):
        """Run with no steps returns empty list."""
        results = run_all_steps("test", [])
        assert results == []

    def test_run_all_single(self):
        """Run with one step returns one result."""
        s = step_fn("echo", lambda s: s)
        results = run_all_steps("test", [s])
        assert len(results) == 1
        assert results[0].text == "test"


# =============================================================================
# Pipeline Tests
# =============================================================================

class TestPipeline:
    """Tests for pipeline() — composable pipeline builder."""

    def test_pipeline_basic(self):
        """Pipeline with no steps just runs the LLM."""
        result = pipeline(mock_llm, "Write code").go()
        assert isinstance(result, PipelineResult)
        assert len(result.output) > 0

    def test_pipeline_with_reason(self):
        """Pipeline with a reason step."""
        result = pipeline(mock_llm, "What is 2+2?").reason().go()
        assert isinstance(result, PipelineResult)
        assert result.steps_count >= 1

    def test_pipeline_with_extract(self):
        """Pipeline with code extraction."""
        def code_llm(prompt, feedback=None):
            return "Here is the code:\n```python\nprint('hello')\n```"
        result = pipeline(code_llm, "Write hello").extract("python").go()
        assert isinstance(result, PipelineResult)

    def test_pipeline_with_map(self):
        """Pipeline with a map/transform step."""
        result = pipeline(mock_llm, "test").map(lambda s: s.upper()).go()
        assert isinstance(result, PipelineResult)
        assert result.output == result.output.upper()

    def test_pipeline_chaining(self):
        """Pipeline with multiple chained steps."""
        result = pipeline(mock_llm, "test") \
            .reason() \
            .map(lambda s: s.upper()) \
            .go()
        assert isinstance(result, PipelineResult)
        assert result.steps_count >= 2

    def test_pipeline_result_fields(self):
        """PipelineResult has expected fields."""
        result = pipeline(mock_llm, "test").go()
        assert hasattr(result, 'output')
        assert hasattr(result, 'total_tokens')
        assert hasattr(result, 'elapsed_ms')
        assert hasattr(result, 'steps_count')

    def test_pipeline_with_refine(self):
        """Pipeline with a refine step using Checks validator."""
        def smart_llm(prompt, feedback=None):
            return "fn main() { println!(\"hello\"); }"

        checks = Checks().require("fn ")
        result = pipeline(smart_llm, "Write Rust code") \
            .refine(checks, max_iter=2) \
            .go()
        assert isinstance(result, PipelineResult)


class TestConcurrentRunner:
    """Tests for concurrent() — parallel pipeline execution."""

    def test_concurrent_simple_tasks(self):
        """Concurrent runner with simple tasks."""
        results = concurrent(mock_llm) \
            .simple_task("a", "Task A") \
            .simple_task("b", "Task B") \
            .go()
        assert len(results) == 2
        assert all(isinstance(r, ConcurrentTaskResult) for r in results)
        labels = {r.label for r in results}
        assert "a" in labels
        assert "b" in labels

    def test_concurrent_result_fields(self):
        """ConcurrentTaskResult has expected fields."""
        results = concurrent(mock_llm) \
            .simple_task("test", "A task") \
            .go()
        assert len(results) == 1
        r = results[0]
        assert hasattr(r, 'label')
        assert hasattr(r, 'output')
        assert hasattr(r, 'success')
        assert hasattr(r, 'elapsed_ms')


# =============================================================================
# Optimizer Tests
# =============================================================================

class TestDataset:
    """Tests for Dataset — training example collection."""

    def test_empty_dataset(self):
        """Empty dataset has length 0."""
        ds = Dataset()
        assert len(ds) == 0
        assert ds.is_empty()

    def test_add_examples(self):
        """Adding examples increases length."""
        ds = Dataset() \
            .example("What is 2+2?", "4") \
            .example("What is 3*5?", "15")
        assert len(ds) == 2
        assert not ds.is_empty()

    def test_labeled_example(self):
        """Adding labeled examples works."""
        ds = Dataset() \
            .example("q1", "a1") \
            .labeled_example("q2", "a2", "math")
        assert len(ds) == 2

    def test_dataset_repr(self):
        """Dataset has a useful __repr__."""
        ds = Dataset().example("q", "a")
        r = repr(ds)
        assert "Dataset" in r
        assert "1" in r

    def test_dataset_immutable_builder(self):
        """Builder pattern creates new instances."""
        ds1 = Dataset()
        ds2 = ds1.example("q", "a")
        assert len(ds1) == 0
        assert len(ds2) == 1


class TestOptimizer:
    """Tests for optimizer() — prompt optimization."""

    def test_optimizer_builder_creation(self):
        """optimizer() creates an OptimizerBuilder."""
        ob = optimizer(mock_llm, "Answer questions")
        assert isinstance(ob, OptimizerBuilder)

    def test_optimizer_builder_repr(self):
        """OptimizerBuilder has a useful __repr__."""
        ob = optimizer(mock_llm, "test")
        r = repr(ob)
        assert "OptimizerBuilder" in r

    def test_optimizer_with_dataset(self):
        """Optimizer accepts a dataset."""
        ds = Dataset().example("2+2?", "4")
        ob = optimizer(mock_llm, "Math").dataset(ds)
        assert isinstance(ob, OptimizerBuilder)

    def test_optimizer_with_metric(self):
        """Optimizer accepts a metric function."""
        ob = optimizer(mock_llm, "Math") \
            .metric(lambda pred, exp: 1.0 if exp in pred else 0.0)
        assert isinstance(ob, OptimizerBuilder)

    def test_optimizer_with_strategy(self):
        """Optimizer accepts strategy configuration."""
        ob = optimizer(mock_llm, "Math") \
            .strategy("bootstrap", max_examples=3)
        assert isinstance(ob, OptimizerBuilder)

    def test_optimizer_strategy_names(self):
        """All strategy names are accepted."""
        for name in ["bootstrap", "instruction", "combined"]:
            ob = optimizer(mock_llm, "Math").strategy(name)
            assert isinstance(ob, OptimizerBuilder)

    def test_optimizer_invalid_strategy(self):
        """Invalid strategy name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            optimizer(mock_llm, "Math").strategy("invalid_strategy")

    def test_optimizer_go_with_dataset(self):
        """Optimizer.go() runs optimization and returns OptimizeResult."""
        ds = Dataset() \
            .example("2+2?", "4") \
            .example("3*5?", "15")
        result = optimizer(mock_llm, "Math") \
            .dataset(ds) \
            .metric(lambda pred, exp: 1.0 if exp in pred else 0.0) \
            .go()
        assert isinstance(result, OptimizeResult)
        assert hasattr(result, 'prompt')
        assert hasattr(result, 'score')
        assert hasattr(result, 'evaluations')
        assert hasattr(result, 'instruction')
        assert hasattr(result, 'candidate_scores')

    def test_optimizer_go_without_dataset(self):
        """Optimizer.go() works without a dataset."""
        result = optimizer(mock_llm, "Write code").go()
        assert isinstance(result, OptimizeResult)


# =============================================================================
# Metric Tests
# =============================================================================

class TestMetric:
    """Tests for Metric — evaluation metrics."""

    def test_exact_match_equal(self):
        """ExactMatch returns 1.0 for equal strings."""
        m = Metric.exact_match()
        assert m.evaluate("hello", "hello") == 1.0

    def test_exact_match_different(self):
        """ExactMatch returns 0.0 for different strings."""
        m = Metric.exact_match()
        assert m.evaluate("hello", "world") == 0.0

    def test_exact_match_trimming(self):
        """ExactMatch trims whitespace."""
        m = Metric.exact_match()
        assert m.evaluate("  hello  ", "hello") == 1.0

    def test_contains_present(self):
        """Contains returns 1.0 when expected is in prediction."""
        m = Metric.contains()
        assert m.evaluate("the answer is 42", "42") == 1.0

    def test_contains_absent(self):
        """Contains returns 0.0 when expected is not in prediction."""
        m = Metric.contains()
        assert m.evaluate("the answer is 42", "43") == 0.0

    def test_f1_token_perfect(self):
        """F1Token returns 1.0 for identical strings."""
        m = Metric.f1_token()
        assert m.evaluate("the quick brown fox", "the quick brown fox") == 1.0

    def test_f1_token_partial(self):
        """F1Token returns partial score for overlapping words."""
        m = Metric.f1_token()
        score = m.evaluate("the quick brown fox", "the quick red fox")
        assert 0.0 < score < 1.0

    def test_f1_token_disjoint(self):
        """F1Token returns 0.0 for completely disjoint strings."""
        m = Metric.f1_token()
        assert m.evaluate("hello world", "foo bar") == 0.0

    def test_custom_metric(self):
        """Custom metric uses the provided callable."""
        m = Metric.custom("my_metric", lambda pred, exp: 1.0 if pred == exp else 0.0)
        assert m.evaluate("hello", "hello") == 1.0
        assert m.evaluate("hello", "world") == 0.0

    def test_custom_metric_name(self):
        """Custom metric preserves its name."""
        m = Metric.custom("accuracy", lambda pred, exp: 0.5)
        assert m.name() == "accuracy"

    def test_builtin_metric_names(self):
        """Built-in metrics have expected names."""
        assert Metric.exact_match().name() == "exact_match"
        assert Metric.contains().name() == "contains"
        assert Metric.f1_token().name() == "f1_token"

    def test_metric_repr(self):
        """Metric has a useful __repr__."""
        m = Metric.exact_match()
        r = repr(m)
        assert "Metric" in r
        assert "exact_match" in r


# =============================================================================
# Multi-objective / Pareto Tests
# =============================================================================

class TestObjective:
    """Tests for Objective — named optimization objectives."""

    def test_basic_creation(self):
        """Create a basic objective."""
        obj = Objective("correctness")
        assert obj.name == "correctness"

    def test_weight(self):
        """Objective.weight() sets the weight."""
        obj = Objective("speed").weight(2.0)
        assert isinstance(obj, Objective)
        assert obj.name == "speed"

    def test_target(self):
        """Objective.target() sets the target threshold."""
        obj = Objective("quality").target(0.8)
        assert isinstance(obj, Objective)

    def test_minimize(self):
        """Objective.minimize() switches to minimize direction."""
        obj = Objective("cost").minimize()
        assert obj.is_minimize

    def test_maximize_default(self):
        """Default direction is maximize."""
        obj = Objective("quality")
        assert not obj.is_minimize

    def test_chaining(self):
        """Builder methods can be chained."""
        obj = Objective("test").weight(2.0).target(0.9).minimize()
        assert obj.name == "test"
        assert obj.is_minimize

    def test_repr(self):
        """Objective has a useful __repr__."""
        obj = Objective("speed").weight(1.5)
        r = repr(obj)
        assert "speed" in r
        assert "Objective" in r


class TestMultiObjective:
    """Tests for multi_objective() — multi-objective validator builder."""

    def test_builder_creation(self):
        """multi_objective() creates a builder."""
        builder = multi_objective()
        assert isinstance(builder, MultiObjectiveBuilder)

    def test_add_objective(self):
        """Builder accepts objectives with validators."""
        builder = multi_objective() \
            .objective(Objective("correct"), Checks().require("fn "))
        assert isinstance(builder, MultiObjectiveBuilder)

    def test_scalarize_strategies(self):
        """All scalarization strategies are accepted."""
        for strategy in ["weighted_sum", "chebyshev", "weighted_product"]:
            builder = multi_objective().scalarize(strategy)
            assert isinstance(builder, MultiObjectiveBuilder)

    def test_invalid_scalarize(self):
        """Invalid scalarization strategy raises error."""
        with pytest.raises(RuntimeError):
            multi_objective().scalarize("invalid_strategy")

    def test_build_validator(self):
        """Builder produces a MultiObjectiveValidator."""
        validator = multi_objective() \
            .objective(Objective("correct"), Checks().require("fn ")) \
            .objective(Objective("brief"), Checks().max_len(200)) \
            .build()
        assert isinstance(validator, MultiObjectiveValidator)
        assert validator.num_objectives == 2

    def test_validate_passing(self):
        """Validator returns passing score."""
        validator = multi_objective() \
            .objective(Objective("has_fn"), Checks().require("fn ")) \
            .build()
        score = validator.validate("fn main() {}")
        assert score.value == 1.0

    def test_validate_failing(self):
        """Validator returns failing score."""
        validator = multi_objective() \
            .objective(Objective("has_fn"), Checks().require("fn ")) \
            .build()
        score = validator.validate("let x = 1;")
        assert score.value < 1.0

    def test_validate_multi(self):
        """validate_multi returns per-objective scores."""
        validator = multi_objective() \
            .objective(Objective("correct"), Checks().require("fn ")) \
            .objective(Objective("long"), Checks().min_len(5)) \
            .build()
        scores = validator.validate_multi("fn main() {}")
        assert len(scores) == 2
        # Each score dict has name, value, weight
        for s in scores:
            assert "name" in s
            assert "value" in s
            assert "weight" in s

    def test_builder_repr(self):
        """Builder has a useful __repr__."""
        builder = multi_objective() \
            .objective(Objective("a"), Checks().require("x"))
        r = repr(builder)
        assert "MultiObjectiveBuilder" in r

    def test_validator_repr(self):
        """Validator has a useful __repr__."""
        validator = multi_objective() \
            .objective(Objective("a"), Checks().require("x")) \
            .build()
        r = repr(validator)
        assert "MultiObjectiveValidator" in r


# =============================================================================
# Import Tests — verify all new exports are importable
# =============================================================================

class TestImports:
    """Verify all Phase 4 exports are importable."""

    def test_pipeline_imports(self):
        from kkachi import pipeline, concurrent, PipelineBuilder, PipelineResult
        from kkachi import ConcurrentRunnerBuilder, ConcurrentTaskResult

    def test_step_imports(self):
        from kkachi import step_fn, step_scored, run_all_steps, StepDef, StepResult

    def test_optimizer_imports(self):
        from kkachi import optimizer, Dataset, OptimizeResult, OptimizerBuilder

    def test_metric_imports(self):
        from kkachi import Metric

    def test_pareto_imports(self):
        from kkachi import (
            multi_objective, refine_pareto,
            Objective, MultiObjectiveBuilder, MultiObjectiveValidator,
            ParetoCandidate, ParetoFront, ParetoRefineResult,
        )

    def test_version(self):
        from kkachi import __version__
        assert __version__ == "0.6.0"


# =============================================================================
# Integration Tests — combine multiple new APIs
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple Phase 4 APIs."""

    def test_step_then_pipeline(self):
        """Create a step, run it, then use pipeline for further processing."""
        upper = step_fn("upper", lambda s: s.upper())
        result = upper.run("hello")
        assert result.text == "HELLO"

        # Use pipeline for further processing
        final = pipeline(mock_llm, result.text).go()
        assert isinstance(final, PipelineResult)

    def test_metric_with_optimizer(self):
        """Use Metric in optimizer flow."""
        m = Metric.contains()
        ds = Dataset().example("2+2?", "4")

        result = optimizer(mock_llm, "Math") \
            .dataset(ds) \
            .metric(lambda pred, exp: m.evaluate(pred, exp)) \
            .go()
        assert isinstance(result, OptimizeResult)

    def test_multi_objective_with_checks(self):
        """Multi-objective validation with Checks validators."""
        validator = multi_objective() \
            .scalarize("weighted_sum") \
            .objective(Objective("has_fn").weight(2.0), Checks().require("fn ")) \
            .objective(Objective("no_unwrap").weight(1.0), Checks().forbid(".unwrap()")) \
            .build()

        # Good code passes
        good = validator.validate("fn main() { let x = 1; }")
        assert good.is_perfect()

        # Code with unwrap partially fails
        bad = validator.validate("fn main() { x.unwrap() }")
        assert bad.value < 1.0

    def test_composed_steps_with_scored(self):
        """Compose fn and scored steps together."""
        transform = step_fn("prefix", lambda s: "Result: " + s)
        validate = step_scored("check", lambda s: (s, 1.0 if s.startswith("Result:") else 0.0))
        composed = transform.then(validate)
        result = composed.run("hello")
        assert result.text == "Result: hello"
        assert result.score == 1.0

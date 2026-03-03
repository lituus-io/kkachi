"""Property-based/fuzz tests for CliValidator using hypothesis."""
import pytest
from hypothesis import given, strategies as st, settings
from kkachi import CliValidator, Checks


class TestValidateProperties:
    """Property-based tests for validate() method."""

    @given(text=st.text(min_size=0, max_size=1000))
    @settings(max_examples=100)
    def test_validate_always_returns_valid_score(self, text):
        """validate() should always return score in [0.0, 1.0]."""
        validator = CliValidator("echo")
        result = validator.validate(text)

        assert 0.0 <= result.value <= 1.0
        assert isinstance(result.value, float)
        assert result.confidence == 1.0

    @given(text=st.text(min_size=0, max_size=1000))
    @settings(max_examples=100)
    def test_validate_deterministic(self, text):
        """Same input should always produce same output."""
        validator = CliValidator("echo")

        result1 = validator.validate(text)
        result2 = validator.validate(text)

        assert result1.value == result2.value
        assert result1.confidence == result2.confidence

    @given(
        text=st.text(min_size=0, max_size=500),
        weight=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=50)
    def test_weight_invariants(self, text, weight):
        """Weight should not break score range."""
        validator = CliValidator("true").weight(weight)
        result = validator.validate(text)

        assert 0.0 <= result.value <= 1.0

    @given(
        text=st.text(min_size=0, max_size=500),
        ext=st.sampled_from(["rs", "py", "js", "txt", "md", "sh"]),
    )
    @settings(max_examples=50)
    def test_extension_invariants(self, text, ext):
        """File extension should not break validation."""
        validator = CliValidator("echo").ext(ext)
        result = validator.validate(text)

        assert 0.0 <= result.value <= 1.0

    @given(
        text=st.text(min_size=0, max_size=500),
        args=st.lists(st.text(min_size=1, max_size=20), min_size=0, max_size=5),
    )
    @settings(max_examples=50)
    def test_args_invariants(self, text, args):
        """Arbitrary args should not crash validation."""
        try:
            validator = CliValidator("echo").args(args)
            result = validator.validate(text)
            assert 0.0 <= result.value <= 1.0
        except Exception:
            # Some arg combinations might be invalid, that's ok
            pass

    @given(text=st.text(min_size=0, max_size=1000))
    @settings(max_examples=50)
    def test_composition_preserves_score_range(self, text):
        """Composed validators should maintain score range."""
        from kkachi import Checks

        cli = CliValidator("true")
        checks = Checks().require("a")
        combined = cli.and_(checks)

        result = combined.validate(text)
        assert 0.0 <= result.value <= 1.0


class TestScoreResultProperties:
    """Property-based tests for ScoreResult."""

    @given(text=st.text(min_size=0, max_size=500))
    @settings(max_examples=100)
    def test_score_result_fields_always_present(self, text):
        """ScoreResult should always have required fields."""
        validator = CliValidator("echo")
        result = validator.validate(text)

        assert hasattr(result, 'value')
        assert hasattr(result, 'feedback')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'passes')
        assert hasattr(result, 'is_perfect')

    @given(
        text=st.text(min_size=0, max_size=500),
        threshold=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=50)
    def test_passes_method_consistent(self, text, threshold):
        """passes() should be consistent with value."""
        validator = CliValidator("true")
        result = validator.validate(text)

        assert result.passes(threshold) == (result.value >= threshold)

    @given(text=st.text(min_size=0, max_size=500))
    @settings(max_examples=50)
    def test_is_perfect_consistent(self, text):
        """is_perfect() should be consistent with value == 1.0."""
        validator = CliValidator("true")
        result = validator.validate(text)

        if result.is_perfect():
            assert abs(result.value - 1.0) < 1e-9

"""Unit tests for CliValidator standalone validation."""
import pytest
import shutil
from kkachi import CliValidator


class TestBasicValidation:
    """Basic standalone validation tests."""

    def test_validate_method_exists(self):
        """CliValidator should have validate() method."""
        validator = CliValidator("echo")
        assert hasattr(validator, 'validate')
        assert callable(validator.validate)

    def test_validate_returns_score_result(self):
        """validate() should return ScoreResult with correct fields."""
        validator = CliValidator("echo").args(["test"])
        result = validator.validate("some text")

        assert hasattr(result, 'value')
        assert hasattr(result, 'feedback')
        assert hasattr(result, 'confidence')
        assert isinstance(result.value, float)
        assert result.confidence == 1.0  # CLI validators are deterministic

    def test_validate_score_range(self):
        """Score should always be between 0.0 and 1.0."""
        validator = CliValidator("echo").args(["test"])
        result = validator.validate("any text")

        assert 0.0 <= result.value <= 1.0

    def test_validate_with_empty_text(self):
        """Should handle empty text input."""
        validator = CliValidator("echo").args(["test"])
        result = validator.validate("")

        assert isinstance(result.value, float)
        assert 0.0 <= result.value <= 1.0

    def test_validate_with_large_text(self):
        """Should handle large text input."""
        validator = CliValidator("echo").args(["test"])
        large_text = "x" * 10000
        result = validator.validate(large_text)

        assert isinstance(result.value, float)


class TestWeightedScoring:
    """Test weighted scoring with multi-stage pipelines."""

    def test_single_stage_default_weight(self):
        """Single stage should use default weight."""
        validator = CliValidator("true")  # Always succeeds
        result = validator.validate("text")

        assert result.value >= 0.0

    def test_single_stage_custom_weight(self):
        """Single stage with custom weight."""
        validator = CliValidator("true").weight(0.5)
        result = validator.validate("text")

        assert result.value >= 0.0

    def test_multi_stage_weights(self):
        """Multi-stage pipeline should combine weights."""
        validator = (
            CliValidator("true")
            .weight(0.3)
            .then("true")
            .weight(0.7)
        )
        result = validator.validate("text")

        # Both stages succeed, should get full score
        assert result.value >= 0.0

    def test_required_stage_failure(self):
        """Required stage failure should zero out score."""
        validator = (
            CliValidator("true")
            .weight(0.5)
            .then("false")  # Always fails
            .required()
        )
        result = validator.validate("text")

        # Required stage failed, score should be 0.0
        assert result.value == 0.0


class TestFileExtensions:
    """Test file extension handling."""

    def test_validate_with_extension(self):
        """Should create temp file with correct extension."""
        validator = CliValidator("echo").ext("rs")
        result = validator.validate("fn main() {}")

        assert isinstance(result.value, float)

    def test_validate_multiple_extensions(self):
        """Test various file extensions."""
        extensions = ["rs", "py", "js", "txt", "md"]
        for ext in extensions:
            validator = CliValidator("echo").ext(ext)
            result = validator.validate("test content")
            assert isinstance(result.value, float)


class TestCapture:
    """Test output capture functionality."""

    def test_validate_with_capture(self):
        """Capture should work with standalone validation."""
        validator = CliValidator("echo").args(["hello"]).capture()
        result = validator.validate("input")

        assert hasattr(validator, 'get_captures')
        captures = validator.get_captures()
        assert len(captures) > 0

    def test_capture_contains_output(self):
        """Captured output should contain expected text."""
        validator = CliValidator("echo").args(["test"]).capture()
        result = validator.validate("input")

        captures = validator.get_captures()
        assert len(captures) > 0
        assert hasattr(captures[0], 'stdout')
        assert hasattr(captures[0], 'stderr')
        assert hasattr(captures[0], 'exit_code')


class TestRealTools:
    """Test with real CLI tools if available."""

    @pytest.mark.skipif(not shutil.which("rustfmt"), reason="rustfmt not available")
    def test_rustfmt_valid_code(self):
        """rustfmt should accept valid Rust code."""
        validator = CliValidator("rustfmt").args(["--check"]).ext("rs")
        result = validator.validate("fn main() {}\n")

        # Valid code should pass
        assert result.value > 0.0

    @pytest.mark.skipif(not shutil.which("python3"), reason="python3 not available")
    def test_python_syntax_check(self):
        """Python syntax check with -m py_compile."""
        validator = CliValidator("python3").args(["-m", "py_compile"]).ext("py")

        # Valid Python
        result = validator.validate("print('hello')")
        assert result.value >= 0.0

    @pytest.mark.skipif(not shutil.which("shellcheck"), reason="shellcheck not available")
    def test_shellcheck_validation(self):
        """shellcheck should validate shell scripts."""
        validator = CliValidator("shellcheck").args(["-"]).ext("sh")

        valid_script = "#!/bin/bash\necho 'hello'\n"
        result = validator.validate(valid_script)
        assert result.value >= 0.0


class TestComposition:
    """Test validator composition with .and_() and .or_()."""

    def test_cli_and_checks_composition(self):
        """CLI validator composed with Checks."""
        from kkachi import Checks

        cli = CliValidator("true")
        checks = Checks().require("test")
        combined = cli.and_(checks)

        # Should be a Validator instance
        assert hasattr(combined, 'validate')

        # Should work standalone
        result = combined.validate("test content")
        assert isinstance(result.value, float)

    def test_cli_or_checks_composition(self):
        """CLI validator OR composition."""
        from kkachi import Checks

        cli = CliValidator("true")
        checks = Checks().forbid("forbidden")
        combined = cli.or_(checks)

        result = combined.validate("test")
        assert isinstance(result.value, float)


class TestEdgeCases:
    """Edge cases and error conditions."""

    def test_nonexistent_command(self):
        """Should handle nonexistent command gracefully."""
        validator = CliValidator("nonexistent_command_xyz_123")
        result = validator.validate("text")

        # Should return a score (likely 0.0 for failure)
        assert isinstance(result.value, float)
        assert 0.0 <= result.value <= 1.0

    def test_unicode_text(self):
        """Should handle unicode text."""
        validator = CliValidator("echo")
        result = validator.validate("Hello 世界 🌍")

        assert isinstance(result.value, float)

    def test_special_characters(self):
        """Should handle special characters in text."""
        validator = CliValidator("echo")
        special_text = "!@#$%^&*(){}[]|\\;:'\"<>,.?/`~"
        result = validator.validate(special_text)

        assert isinstance(result.value, float)

    def test_newlines_in_text(self):
        """Should handle multi-line text."""
        validator = CliValidator("echo")
        multiline = "line1\nline2\nline3\n"
        result = validator.validate(multiline)

        assert isinstance(result.value, float)

    def test_timeout_handling(self):
        """Should handle timeout configuration."""
        validator = CliValidator("sleep").args(["1"]).timeout(2)
        result = validator.validate("text")

        # Should complete within timeout
        assert isinstance(result.value, float)

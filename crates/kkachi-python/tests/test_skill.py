# Copyright © 2025 lituus-io <spicyzhug@gmail.com>
# All Rights Reserved.
# Licensed under PolyForm Noncommercial 1.0.0

"""Tests for the Skill type across all DSPy modules."""

import pytest
from kkachi import Skill, Defaults, reason, best_of, ensemble, agent, program


def test_skill_basic():
    """Create a skill and add instructions."""
    skill = Skill().instruct("naming", "Use snake_case.").instruct("policy", "No deletionProtection.")
    assert len(skill) == 2
    assert not skill.is_empty()


def test_skill_render():
    """Render output should contain instructions."""
    skill = Skill().instruct("naming", "Use snake_case for all names.")
    rendered = skill.render()
    assert "## Instructions" in rendered
    assert "naming" in rendered
    assert "snake_case" in rendered


def test_skill_with_reason():
    """Skill integrates with reason() builder."""
    def fake_llm(prompt, feedback=None):
        if "## Instructions" in prompt and "snake_case" in prompt:
            return "Answer: skill works"
        return "Answer: no skill"

    skill = Skill().instruct("naming", "Use snake_case.")
    result = reason(fake_llm, "Generate code").skill(skill).go()
    assert "skill works" in result.output


def test_skill_empty():
    """Empty skill works without error."""
    skill = Skill()
    assert len(skill) == 0
    assert skill.is_empty()
    assert skill.render() == ""


def test_skill_repr():
    """Repr should show instruction count."""
    skill = Skill().instruct("a", "b").instruct("c", "d")
    r = repr(skill)
    assert "Skill" in r
    assert "2" in r


def test_skill_priority():
    """Instructions with explicit priority should be ordered."""
    skill = (
        Skill()
        .instruct_at("low", "Low priority.", 200)
        .instruct_at("high", "High priority.", 10)
    )
    rendered = skill.render()
    high_pos = rendered.index("high")
    low_pos = rendered.index("low")
    assert high_pos < low_pos


def test_skill_chaining():
    """Builder pattern returns new instances."""
    s1 = Skill()
    s2 = s1.instruct("a", "b")
    s3 = s2.instruct("c", "d")
    assert len(s1) == 0
    assert len(s2) == 1
    assert len(s3) == 2


# =============================================================================
# Skill + Defaults with best_of()
# =============================================================================


def test_skill_with_best_of():
    """Skill integrates with best_of() builder."""
    def fake_llm(prompt, feedback=None):
        if "## Instructions" in prompt and "snake_case" in prompt:
            return "skill applied"
        return "no skill"

    skill = Skill().instruct("naming", "Use snake_case.")
    result = best_of(fake_llm, "Generate code", 1).skill(skill).go()
    assert "skill applied" in result.output


def test_defaults_with_best_of():
    """Defaults integrates with best_of() builder."""
    def fake_llm(prompt, feedback=None):
        return "user:admin@example.com"

    defaults = Defaults().set("email", r"admin@example\.com", "real@company.com")
    result = best_of(fake_llm, "Generate IAM", 1).defaults(defaults).go()
    assert "real@company.com" in result.output
    assert "admin@example.com" not in result.output


# =============================================================================
# Skill + Defaults with ensemble()
# =============================================================================


def test_skill_with_ensemble():
    """Skill integrates with ensemble() builder."""
    def fake_llm(prompt, feedback=None):
        if "## Instructions" in prompt and "snake_case" in prompt:
            return "skill applied"
        return "no skill"

    skill = Skill().instruct("naming", "Use snake_case.")
    result = ensemble(fake_llm, "Generate code", 1).skill(skill).go()
    assert "skill applied" in result.output


def test_defaults_with_ensemble():
    """Defaults integrates with ensemble() builder."""
    def fake_llm(prompt, feedback=None):
        return "user:admin@example.com"

    defaults = Defaults().set("email", r"admin@example\.com", "real@company.com")
    result = ensemble(fake_llm, "Generate IAM", 1).defaults(defaults).go()
    assert "real@company.com" in result.output
    assert "admin@example.com" not in result.output


# =============================================================================
# Skill + Defaults with agent()
# =============================================================================


def test_skill_with_agent():
    """Skill integrates with agent() builder."""
    def fake_llm(prompt, feedback=None):
        if "## Instructions" in prompt and "snake_case" in prompt:
            return "Final Answer: skill applied"
        return "Final Answer: no skill"

    skill = Skill().instruct("naming", "Use snake_case.")
    result = agent(fake_llm, "Generate code").skill(skill).go()
    assert result.output == "skill applied"


def test_defaults_with_agent():
    """Defaults integrates with agent() builder."""
    def fake_llm(prompt, feedback=None):
        return "Final Answer: admin@example.com"

    defaults = Defaults().set("email", r"admin@example\.com", "real@company.com")
    result = agent(fake_llm, "Get email").defaults(defaults).go()
    assert result.output == "real@company.com"


# =============================================================================
# Skill + Defaults with program()
# =============================================================================


def test_skill_with_program():
    """Skill integrates with program() builder."""
    from kkachi import Executor

    def fake_llm(prompt, feedback=None):
        if "## Instructions" in prompt and "snake_case" in prompt:
            return "```bash\necho skill_applied\n```"
        return "```bash\necho no_skill\n```"

    skill = Skill().instruct("naming", "Use snake_case.")
    result = program(fake_llm, "Generate code").executor(Executor.bash()).skill(skill).go()
    assert result.success
    assert "skill_applied" in result.output


def test_defaults_with_program():
    """Defaults integrates with program() builder."""
    from kkachi import Executor

    def fake_llm(prompt, feedback=None):
        return "```bash\necho admin@example.com\n```"

    defaults = Defaults().set("email", r"admin@example\.com", "real@company.com")
    result = program(fake_llm, "Generate IAM").executor(Executor.bash()).defaults(defaults).go()
    assert result.success
    assert "real@company.com" in result.output
    assert "admin@example.com" not in result.output

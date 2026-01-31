#!/usr/bin/env python3
"""Revert consuming pattern back to cloning pattern (PyO3 requirement)."""

import re

# Read the file
with open('src/dspy.rs', 'r') as f:
    content = f.read()

# Add back Clone impls for the 4 builders
builders_needing_clone = [
    ('PyRefineBuilder', '''impl Clone for PyRefineBuilder {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            llm: self.llm.clone_ref(py),
            prompt: self.prompt.clone(),
            max_iter: self.max_iter,
            target: self.target,
            require_patterns: self.require_patterns.clone(),
            forbid_patterns: self.forbid_patterns.clone(),
            validator: self.validator.clone(),
        })
    }
}

'''),
    ('PyReasonBuilder', '''impl Clone for PyReasonBuilder {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            llm: self.llm.clone_ref(py),
            query: self.query.clone(),
            max_iter: self.max_iter,
            target: self.target,
            include_reasoning: self.include_reasoning,
            require_patterns: self.require_patterns.clone(),
            forbid_patterns: self.forbid_patterns.clone(),
            validator: self.validator.clone(),
        })
    }
}

'''),
    ('PyBestOfBuilder', '''impl Clone for PyBestOfBuilder {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            llm: self.llm.clone_ref(py),
            prompt: self.prompt.clone(),
            n: self.n,
            scorer: self.scorer.as_ref().map(|s| s.clone_ref(py)),
            scorer_weight: self.scorer_weight,
            require_patterns: self.require_patterns.clone(),
            forbid_patterns: self.forbid_patterns.clone(),
            validator: self.validator.clone(),
        })
    }
}

'''),
    ('PyEnsembleBuilder', '''impl Clone for PyEnsembleBuilder {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            llm: self.llm.clone_ref(py),
            question: self.question.clone(),
            n: self.n,
            aggregate: self.aggregate.clone(),
            normalize: self.normalize,
            with_reasoning: self.with_reasoning,
            require_patterns: self.require_patterns.clone(),
            forbid_patterns: self.forbid_patterns.clone(),
            validator: self.validator.clone(),
        })
    }
}

'''),
]

for builder_name, clone_impl in builders_needing_clone:
    # Find where to insert (just before #[pymethods])
    pattern = rf'(#\[pyclass\(name = "{builder_name.replace("Py", "")}"\)\]\npub struct {builder_name} \{{[^}}]+\}})\n\n(#\[pymethods\])'
    match = re.search(pattern, content)
    if match:
        content = content[:match.end(1)] + '\n\n' + clone_impl + match.group(2) + content[match.end(2):]
        print(f"Added Clone impl for {builder_name}")

# Replace mut self -> &self
content = re.sub(
    r'fn (validate|max_iter|target|scorer_weight|with_reasoning|aggregate|no_normalize|no_reasoning|require|forbid|regex|metric|tool|max_steps|executor|max_len|min_len|no_code|language|adaptive|with_budget)\(mut self,',
    r'fn \1(&self,',
    content
)

# Add clone lines back
for method in ['validate', 'max_iter', 'target', 'require', 'forbid', 'metric', 'scorer_weight',
               'with_reasoning', 'aggregate', 'no_normalize', 'no_reasoning', 'regex', 'tool',
               'max_steps', 'executor', 'no_code', 'language']:
    # Pattern: method(&self, ...) -> ... {\n        self.field = value;
    # Replace with: method(&self, ...) -> ... {\n        let mut new = self.clone();\n        new.field = value;
    content = re.sub(
        rf'(fn {method}\(&self,[^{{]*\{{)\n        (self\.\w+)',
        r'\1\n        let mut new = self.clone();\n        new.\2',
        content
    )

# Change self.field -> new.field (where preceded by "let mut new = self.clone();")
content = re.sub(r'(let mut new = self\.clone\(\);\n        )self\.', r'\1new.', content)

# Continue replacing self. with new. in subsequent lines of the same method
lines = content.split('\n')
in_builder_method = False
for i, line in enumerate(lines):
    if 'let mut new = self.clone();' in line:
        in_builder_method = True
    elif in_builder_method:
        if line.strip().startswith('self.') and not line.strip().startswith('self,'):
            lines[i] = line.replace('self.', 'new.')
        if line.strip() in ['self', 'Ok(self)', 'new', 'Ok(new)']:
            # Replace return self with return new
            if 'self' in line.strip():
                lines[i] = line.replace('self', 'new')
            in_builder_method = False

content = '\n'.join(lines)

# Write the result
with open('src/dspy.rs', 'w') as f:
    f.write(content)

print("Reverted to cloning pattern (PyO3 compatible)")

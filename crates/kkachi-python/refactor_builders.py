#!/usr/bin/env python3
"""Refactor DSPy builders to use consuming pattern (zero-copy)."""

import re

# Read the file
with open('src/dspy.rs', 'r') as f:
    content = f.read()

# Remove Clone impls for the 4 remaining builders
for builder in ['PyBestOfBuilder', 'PyEnsembleBuilder', 'PyAgentBuilder', 'PyProgramBuilder']:
    # Find and remove the impl Clone block
    pattern = rf'impl Clone for {builder} \{{\n(?:.*?\n)*?\}}\n\n'
    content = re.sub(pattern, '', content, count=1)
    print(f"Removed Clone impl for {builder}")

# Replace cloning builder methods with consuming pattern
# Pattern: fn method_name(&self, ...) -> Self { let mut new = self.clone(); new.field = value; new }
# Replace with: fn method_name(mut self, ...) -> Self { self.field = value; self }

patterns_to_fix = [
    # &self -> mut self
    (r'fn (validate|max_iter|target|scorer_weight|with_reasoning|aggregate|no_normalize|no_reasoning|'
     r'require|forbid|regex|metric|tool|max_steps|executor|max_len|min_len|no_code|language|adaptive|with_budget)\(&self,',
     r'fn \1(mut self,'),

    # Remove clone lines
    (r'\s+let mut new = self\.clone\(\);\n', ''),

    # new.field -> self.field
    (r'\bnew\.(\w+) =', r'self.\1 ='),
    (r'\bnew\.(\w+)\.push\(', r'self.\1.push('),

    # return new -> return self
    (r'\n\s+new\n\s+\}', '\n        self\n    }'),
    (r'Ok\(new\)', 'Ok(self)'),
]

for pattern, replacement in patterns_to_fix:
    content = re.sub(pattern, replacement, content)

# Special fix for metric method (has Python::with_gil)
content = re.sub(
    r'fn metric\(mut self, scorer: PyObject\) -> Self \{\s*Python::with_gil\(\|py\| \{\s*self\.scorer = Some\(scorer\.clone_ref\(py\)\);\s*\}\);\s*self\s*\}',
    '''fn metric(mut self, scorer: PyObject) -> Self {
        self.scorer = Python::with_gil(|py| Some(scorer.clone_ref(py)));
        self
    }''',
    content
)

# Write the result
with open('src/dspy.rs', 'w') as f:
    f.write(content)

print("Refactoring complete!")

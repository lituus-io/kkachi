# kkachi-python

Python bindings for the [kkachi](https://github.com/lituus-io/kkachi) LLM optimization library.

## Installation

```bash
pip install kkachi
```

## Usage

```python
from kkachi import Signature, Example

sig = Signature("question -> answer")
ex = Example({"question": "What is 2+2?", "answer": "4"})
```

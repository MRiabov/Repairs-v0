---
trigger: model_decision
description: When creating new class variable
---

It is preferable to create class variable docstrings instead of comments. E.g:

```py
class Class123:
    var1: int
    """Variable description"""
```
is preferred over 
```py
class Class123:
    # Variable description
    var1: int
```

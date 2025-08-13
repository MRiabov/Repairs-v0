---
trigger: model_decision
description: When deciding which tests to run
---

When calling `pytest`, it's better to never execute the whole test suite because it takes over 5 minutes to run. So, never run `pytest -q` without test file or `-k`. Instead, run an individual test function, class, module or a combination of them.
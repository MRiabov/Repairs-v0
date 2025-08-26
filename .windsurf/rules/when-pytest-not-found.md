---
trigger: model_decision
description: When pytest is not found
---

Sometimes when running tests, `pytest` would be not found. This means that `venv` is not activated, and you can activate it with `source ../.venv/bin/activate`. After that, it will certainly work. Do not try to run `./venv/bin/pytest` directly, activate the venv, and then run `pytest` as usual.
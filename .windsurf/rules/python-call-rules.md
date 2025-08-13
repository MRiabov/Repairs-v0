---
trigger: model_decision
description: When calling `python`, `pip` or `pytest` in console
---

1. When calling python or python modules (pip or pytest), use python3 -m [module] instead of `python`.
2. When calling `pip` use `uv pip [command]` instead of `pip`; 
3. When calling pytest you can just use `pytest [args]`.
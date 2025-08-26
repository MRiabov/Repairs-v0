---
trigger: always_on
---

When deciding if to write defensive logic, e.g. dimensionality handling: `tensor1=tensor1.unsqueeze() if tensor1.ndim==1 else tensor1`, or None handling: `var1=var1 if var1 else torch.zeros(...)`, just don't write these things. In my code, shapes are always static, and there is one execution path for all code. I prefer `assert` over defensive logic. If you are writing something to fix the tests and this seems necessary, it's likely that the tests are setup incorrectly.
The reason why I don't want it is because defensive logic leads to silent failures, and these are bad for debugging.

In addition, writing "int()" type casting is also a piece of defensive logic that slows down the application. Don't write it unless really necessary e.g. putting int to string. In most cases, indexing with a tensor should be better.
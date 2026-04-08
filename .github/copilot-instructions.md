---
name: 'Python Standards'
description: 'Coding conventions for Python files'
applyTo: '**/*.py'
---

In this workspace, use snake case and other typical Python conventions for casing.

In this workspace, use 4 spaces for indentation and not tabs.

When writing a docstring, if the function (or other object) involves the torch or einops packages, then write a section with that package as the name. Discuss the function as it relates to the other package. Write the section in the style of the other package.

## Development goals

1. Code changes should not require lucidrains, the developer, to change their coding style.
   1. They don't use type annotations very often, so the code for typing should be unobtrusive.
2. Confirm that changes do not break packages that import these symbols.
3. Use formatters and linters to make new code match the style of the existing codebase.

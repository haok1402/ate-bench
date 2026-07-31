# Judge: {task}

{intro}

**The diff:** `{patches}` holds one patch file per modified codebase. Read every patch file there, carefully and end-to-end. Judge only what the diff actually contains: mark FAIL when the change is absent, even if a commit message or comment claims otherwise.

## Rule

{rule}

## Output format

Output exactly the following and stop. Do not add any other commentary, preamble, or summary, and do not wrap it in a code fence.

```
<PASS|FAIL>

<justification>
```

The verdict is the single word `PASS` or `FAIL` on the first line. The justification is a single paragraph. Ground it in the diff: name the files, symbols, and code paths you relied on, say what they do, and explain why that does or does not satisfy the rule. Where the evidence is partial or ambiguous, state which part is missing and how you resolved it.

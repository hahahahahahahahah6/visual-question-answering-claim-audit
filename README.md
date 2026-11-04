# Documentation claim auditor

An offline checker for documentation claims about answering questions grounded in pictures. It compares exact prose phrases with static source patterns. Runtime inspection uses only the standard library, with no network access, telemetry or file writes.

## Usage

Requires Python 3.9 or newer.

```sh
python3 -m pip install visual-question-answering-claim-audit
visual-question-answering-claim-audit --version
visual-question-answering-claim-audit audit .
python3 -m visual_question_answering_claim_audit audit . --json
visual-question-answering-claim-audit audit . --verbose --ignore vendor/ --ignore build/ --fail-on medium
```

The directory defaults to the current directory. The scanner reads root documentation and Markdown beneath the documentation directory, plus Markdown and source fixtures beneath test directories. Source evidence is searched recursively, skipping version control, caches and virtual environments. Repeatable path prefixes exclude both documentation and sources.

```text
README.md
docs/**/*.md
tests/**/*.md
tests/**/*.py
*.py
```

Fenced code is excluded from prose; inline code counts. Matching ignores case, preserves hyphens and removes longer matching spans before shorter ones. Denial spans are removed before positive matching.

JSON output is a list of findings with rule, verdict, message, path and phrase fields. Verdict denotes HIGH or MEDIUM severity. Verbose output describes scan scope and per-file decisions on standard error. Status 0 means no findings meet the threshold; status 2 means findings meet it; status 1 means a usage or input error. The default threshold is high. Test-only positive claims receive MEDIUM severity and fail only when medium is selected.

```text
documented-visual-question-answering-claim-false
documented-no-visual-question-answering-claim-false
```

## Source patterns

Imports and aliases resolve within each module. Local function and class definitions conservatively shadow route names throughout that module. A supported task call, a supported model construction paired with an instance call, or a listed generic call establishes evidence. Imports alone, strings, comments and uncalled references do not.

```python
from transformers import pipeline as make
engine = make(task="visual-question-answering")
```

The paired routes require generation for the first model below, and invocation or generation for the other two. Generic attribute calls accept any base; bare calls require an imported name. Separate modules cannot complete a pair.

```text
BlipForQuestionAnswering
ViltForQuestionAnswering
LxmertForQuestionAnswering
```

## Differentiation

The neighboring checkers inspect different outputs:

```text
image-captioning-claim-audit: scene-description sentences
question-answering-claim-audit: text-only answers
ocr-claim-audit: transcripts of embedded text
```

This checker concerns answers to specific questions grounded in pictures. A scene-description sentence, an answer grounded only in text, or a transcript of embedded lettering does not establish its source routes. Classification labels, bounding boxes, segmentation masks and generated pictures also do not suffice.

## Limitations

Static pattern matching is not semantic judgement and does not prove runtime behavior or answer quality. Exact phrases miss paraphrases. Generic names can cause false positives. Dynamic dispatch, wrappers and cross-module data flow can be missed. Sources are parsed, never executed; malformed or unreadable scanned files produce an input error.

## Development

```sh
python3 -m unittest discover
python3 -m build
```

The source archive includes the behavioral test suite. Distributed under the MIT license.

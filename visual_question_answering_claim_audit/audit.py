"""Deterministic prose matching and module-local AST inspection."""
import ast
from pathlib import Path
import re

from .phrases import VQA, NOVQA, GENERIC_NAMES


def prose(text):
    lines = []
    fence = None
    for line in text.splitlines(keepends=True):
        match = re.match(r'^\s*(`{3,}|~{3,})', line)
        if match:
            marker = match.group(1)
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
            lines.append('\n')
        elif fence is None:
            lines.append(line.replace('`', ''))
    return ''.join(lines)


def phrase_matches(text, phrases):
    text = text.lower()
    found = []
    for phrase in sorted(phrases, key=len, reverse=True):
        start = text.find(phrase)
        while start >= 0:
            found.append(phrase)
            text = text[:start] + ' ' * len(phrase) + text[start + len(phrase):]
            start = text.find(phrase)
    return found, text


def has_evidence(source):
    tree = ast.parse(source)
    nodes = list(ast.walk(tree))
    shadow = {n.name for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    imports = {}
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports[alias.asname or alias.name.split('.')[0]] = alias.name if alias.asname else alias.name.split('.')[0]
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                imports[alias.asname or alias.name] = node.module + '.' + alias.name

    def resolve(node):
        if isinstance(node, ast.Name):
            return '' if node.id in shadow else imports.get(node.id, '')
        if isinstance(node, ast.Attribute):
            base = resolve(node.value)
            return base + '.' + node.attr if base and node.attr not in shadow else ''
        return ''

    models = {'BlipForQuestionAnswering': 'generate', 'ViltForQuestionAnswering': 'call',
              'LxmertForQuestionAnswering': 'call'}

    def construction(node):
        if not isinstance(node, ast.Call):
            return None
        name = resolve(node.func)
        parts = name.split('.')
        if parts[-1] == 'from_pretrained':
            parts.pop()
        if parts and parts[0] == 'transformers' and parts[-1] in models:
            return parts[-1]
        return None

    instances = {}
    for node in nodes:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            kind = construction(node.value)
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if kind:
                for target in targets:
                    if isinstance(target, ast.Name) and target.id not in shadow:
                        instances[target.id] = kind
    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        resolved = resolve(func)
        if resolved == 'transformers.pipeline':
            values = ([node.args[0]] if node.args else []) + [k.value for k in node.keywords if k.arg == 'task']
            if any(isinstance(v, ast.Constant) and v.value == 'visual-question-answering' for v in values):
                return True
        if isinstance(func, ast.Attribute) and func.attr in GENERIC_NAMES and func.attr not in shadow:
            if not isinstance(func.value, ast.Name) or func.value.id not in shadow:
                return True
        if isinstance(func, ast.Name) and resolved and resolved.split('.')[-1] in GENERIC_NAMES:
            return True
        if isinstance(func, ast.Name) and instances.get(func.id) in ('ViltForQuestionAnswering', 'LxmertForQuestionAnswering'):
            return True
        if isinstance(func, ast.Attribute) and func.attr == 'generate':
            if isinstance(func.value, ast.Name) and func.value.id in instances:
                return True
            if construction(func.value):
                return True
        if isinstance(func, ast.Call) and construction(func) in ('ViltForQuestionAnswering', 'LxmertForQuestionAnswering'):
            return True
    return False


EXCLUDED = {'.git', '__pycache__', '.venv', 'venv', 'env', '.env'}
POSITIVE_RULE = 'documented-visual-question-answering-claim-false'
NEGATIVE_RULE = 'documented-no-visual-question-answering-claim-false'


def audit(repo, ignore=()):
    root = Path(repo)
    if not root.is_dir():
        raise OSError('Repository is not a directory: ' + str(root))
    decisions = []
    documents = []
    evidence = False
    paths = sorted(root.rglob('*'))
    venvs = {p.parent for p in paths if p.name == 'pyvenv.cfg'}
    for path in paths:
        rel = path.relative_to(root)
        name = rel.as_posix()
        if any(part in EXCLUDED for part in rel.parts) or any(v in path.parents for v in venvs) or any(name.startswith(prefix) for prefix in ignore):
            continue
        if not path.is_file():
            continue
        test = 'tests' in rel.parts[:-1]
        doc = name == 'README.md' or (rel.parts[0] == 'docs' and path.suffix == '.md')
        if path.suffix == '.py' or doc or (test and path.suffix == '.md'):
            text = path.read_text(encoding='utf-8')
            if path.suffix == '.py':
                hit = has_evidence(text)
                evidence = evidence or hit
                decisions.append(name + ': evidence=' + str(hit))
            if doc or (test and path.suffix in ('.py', '.md')):
                documents.append((name, prose(text), test))
    findings = []
    for name, text, test in documents:
        denials, remaining = phrase_matches(text, NOVQA)
        claims, _ = phrase_matches(remaining, VQA)
        decisions.append(name + ': claims=' + str(len(claims)) + ', denials=' + str(len(denials)))
        for phrase in claims if not evidence else []:
            findings.append(dict(rule=POSITIVE_RULE, verdict='MEDIUM' if test else 'HIGH', message='Documented claim has no recognized source evidence.', path=name, phrase=phrase))
        for phrase in denials if evidence else []:
            findings.append(dict(rule=NEGATIVE_RULE, verdict='HIGH', message='Documented denial conflicts with recognized source evidence.', path=name, phrase=phrase))
    return findings, decisions

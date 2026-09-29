"""Check repository structure without reading private datasets or running notebooks."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import sys
import tomllib
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def check_repository(root: Path = ROOT) -> list[str]:
    errors = []
    config = tomllib.loads((root / 'pyproject.toml').read_text(encoding='utf-8'))
    if config['tool']['setuptools']['packages']['find']['where'] != ['src']:
        errors.append('Package discovery must use the src layout.')
    if not (root / 'src/catalogiq/__init__.py').is_file():
        errors.append('Missing catalogiq package.')

    for directory in ['src', 'tests', 'scripts']:
        for path in (root / directory).rglob('*.py'):
            try:
                ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
            except SyntaxError as error:
                errors.append(f'{path.relative_to(root)}: {error}')

    notebook_index = (root / 'notebooks/README.md').read_text(encoding='utf-8')
    for path in (root / 'notebooks').glob('*.ipynb'):
        try:
            notebook = json.loads(path.read_text(encoding='utf-8'))
            assert notebook['nbformat'] == 4
            assert isinstance(notebook['cells'], list)
            for cell in notebook['cells']:
                assert cell['cell_type'] in {'code', 'markdown', 'raw'}
                assert isinstance(cell['source'], (list, str))
                if cell['cell_type'] == 'code':
                    assert isinstance(cell['outputs'], list)
            if f'({path.name})' not in notebook_index:
                errors.append(f'{path.name}: missing from notebook index.')
        except (ValueError, KeyError, AssertionError, TypeError) as error:
            errors.append(f'{path.name}: invalid notebook structure: {error}')

    markdown = [root / 'README.md', root / 'CONTRIBUTING.md']
    for directory in ['docs', 'notebooks', 'data']:
        # Data contains local outputs; only check its top-level guide.
        markdown.extend((root / directory).rglob('*.md') if directory != 'data'
                        else (root / directory).glob('*.md'))
    for path in markdown:
        text = path.read_text(encoding='utf-8')
        # Exclude fenced code examples; only check actual inline Markdown links.
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        for target in re.findall(r'\[[^\]]+\]\(([^\n]+?)\)', text):
            target = target.strip().strip('<>')
            if target.startswith('#') or urlsplit(target).scheme:
                continue
            local_path = unquote(target.split('#')[0])
            if local_path and not (path.parent / local_path).exists():
                errors.append(f'{path.relative_to(root)}: broken link {target}')
    return errors


if __name__ == '__main__':
    problems = check_repository()
    if problems:
        print('\n'.join(problems), file=sys.stderr)
        raise SystemExit(1)
    print('Repository checks passed: package layout, Python syntax, notebook structure/index, local documentation links.')

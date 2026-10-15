"""Git clean filter: strip execution data in Git, preserving local notebooks."""
import json
import sys


def clean(notebook):
    notebook['metadata'] = {key: value for key, value in notebook.get('metadata', {}).items()
                            if key in ('kernelspec', 'language_info')}
    for cell in notebook['cells']:
        cell['metadata'] = {}
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
    return notebook


if __name__ == '__main__':
    json.dump(clean(json.load(sys.stdin)), sys.stdout, indent=1, ensure_ascii=False)
    sys.stdout.write('\n')

"""Run a registered command: python -m scripts <name> [arguments]."""
import runpy
import sys
from scripts._paths import SCRIPT_PATHS, resolve_script


def main():
    if len(sys.argv) == 1 or sys.argv[1] in ('--list', '--help', '-h'):
        print('Usage: python -m scripts <command> [arguments]\n')
        for name, path in sorted(SCRIPT_PATHS.items(), key=lambda x: x[1]):
            print(f'{name[:-3]:38} scripts/{path}')
        return
    name = sys.argv[1].removesuffix('.py') + '.py'
    if name not in SCRIPT_PATHS:
        raise SystemExit(f'Unknown command: {sys.argv[1]}. Use python -m scripts --list.')
    path = resolve_script(name)
    sys.argv = [str(path), *sys.argv[2:]]
    runpy.run_path(str(path), run_name='__main__')


if __name__ == '__main__':
    main()

"""Command-line interface."""
import argparse
import json
import sys
from . import __version__
from .audit import audit


class Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(1, self.prog + ': error: ' + message + '\n')


def main(argv=None):
    parser = Parser()
    parser.add_argument('--version', action='version', version=__version__)
    commands = parser.add_subparsers(dest='command', required=True)
    scan = commands.add_parser('audit')
    scan.add_argument('repo', nargs='?', default='.')
    scan.add_argument('--json', action='store_true')
    scan.add_argument('--verbose', action='store_true')
    scan.add_argument('--ignore', action='append', nargs='+', default=[])
    scan.add_argument('--fail-on', choices=('high', 'medium'), default='high')
    args = parser.parse_args(argv)
    try:
        findings, decisions = audit(args.repo, [p for group in args.ignore for p in group])
    except (OSError, UnicodeError, SyntaxError) as error:
        print(str(error), file=sys.stderr)
        return 1
    if args.verbose:
        print('Scan scope: ' + args.repo, file=sys.stderr)
        print('\n'.join(decisions), file=sys.stderr)
    if args.json:
        print(json.dumps(findings, indent=2))
    elif findings:
        for finding in findings:
            print('{verdict} {rule} {path}: {message}'.format(**finding))
    else:
        print('PASS')
    return 2 if any(f['verdict'] == 'HIGH' or args.fail_on == 'medium' for f in findings) else 0


if __name__ == '__main__':
    sys.exit(main())

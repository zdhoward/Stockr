import argparse
from pprint import pprint

def main(command_line=None):
    args = parse_arguments(command_line)
    pprint(args)

def parse_arguments(command_line=None):
    parser = argparse.ArgumentParser('STOCKR')
    parser.add_argument("--verbose", action="store_true")

    module_parser = parser.add_subparsers(dest='module')

    predict_parser = module_parser.add_parser('predict', help='predict prices')
    predict_parser.add_argument("--tickers")
    predict_parser.add_argument("--portfolio")
    predict_parser.add_argument("--open-folders")

    portfolio_parser = module_parser.add_parser('portfolio', help='manage portfolio')
    portfolio_parser.add_argument("set")
    portfolio_parser.add_argument("view")
    portfolio_parser.add_argument("add")
    portfolio_parser.add_argument("remove")


    args = parser.parse_args(command_line)

    return args

def parse_old():
    parser = argparse.ArgumentParser('Blame Praise app')
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Print debug info'
    )
    subprasers = parser.add_subparsers(dest='command')
    blame = subprasers.add_parser('blame', help='blame people')
    blame.add_argument(
        '--dry-run',
        help='do not blame, just pretend',
        action='store_true'
    )
    blame.add_argument('name', nargs='+', help='name(s) to blame')
    praise = subprasers.add_parser('praise', help='praise someone')
    praise.add_argument('name', help='name of person to praise')
    praise.add_argument(
        'reason',
        help='what to praise for (optional)',
        default="no reason",
        nargs='?'
    )
    args = parser.parse_args(command_line)
    if args.debug:
        print("debug: " + str(args))
    if args.command == 'blame':
        if args.dry_run:
            print("Not for real")
        print("blaming " + ", ".join(args.name))
    elif args.command == 'praise':
        print('praising ' + args.name + ' for ' + args.reason)

if __name__=="__main__":
    main()
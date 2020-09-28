import argparse
from pprint import pprint
import os
preference_folder = "user/"
import configparser
def main(command_line=None):
    args = parse_arguments(command_line)
    pprint(args)
    print(args.set)
    prefs = {'stocks': args.set}
    save_prefs(prefs)


def parse_arguments(command_line=None):
    parser = argparse.ArgumentParser("STOCKR")
    parser.add_argument("-v", "--verbose", action="store_true")

    module_parser = parser.add_subparsers(dest="module")

    predict_parser = module_parser.add_parser("predict", help="predict prices")
    predict_parser.add_argument("-t", "--tickers")
    predict_parser.add_argument("-p", "--portfolio", action='store_true')
    predict_parser.add_argument("-o", "--open-folders", action='store_true')

    portfolio_parser = module_parser.add_parser("portfolio", help="manage portfolio")
    portfolio_parser.add_argument("-s", "--set")
    portfolio_parser.add_argument("-v", "--view", action="store_true")
    portfolio_parser.add_argument("-a", "--add")
    portfolio_parser.add_argument("-r", "--remove")

    args = parser.parse_args(command_line)

    return args

def save_prefs(prefs):
    dir = ".config"
    if not os.path.exists(dir):
        os.mkdir(dir)
    Config = configparser.ConfigParser()
    cfgfile = open(os.path.join(dir, 'stockr.cfg'), 'w')
    Config.add_section('portfolio')
    Config.set('portfolio', 'stocks', prefs['stocks'])
    Config.write(cfgfile)
    cfgfile.close()
    

def parse_old():
    parser = argparse.ArgumentParser("Blame Praise app")
    parser.add_argument("--debug", action="store_true", help="Print debug info")
    subprasers = parser.add_subparsers(dest="command")
    blame = subprasers.add_parser("blame", help="blame people")
    blame.add_argument(
        "--dry-run", help="do not blame, just pretend", action="store_true"
    )
    blame.add_argument("name", nargs="+", help="name(s) to blame")
    praise = subprasers.add_parser("praise", help="praise someone")
    praise.add_argument("name", help="name of person to praise")
    praise.add_argument(
        "reason", help="what to praise for (optional)", default="no reason", nargs="?"
    )
    args = parser.parse_args(command_line)
    if args.debug:
        print("debug: " + str(args))
    if args.command == "blame":
        if args.dry_run:
            print("Not for real")
        print("blaming " + ", ".join(args.name))
    elif args.command == "praise":
        print("praising " + args.name + " for " + args.reason)


if __name__ == "__main__":
    main()
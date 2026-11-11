import sys
from cwnn.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["fit", *sys.argv[1:]]))

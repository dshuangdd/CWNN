import sys

from run import main

if __name__ == "__main__":
    main(["--example", "1", *sys.argv[1:]])

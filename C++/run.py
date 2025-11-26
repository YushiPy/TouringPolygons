
import sys


def main(args: list[str]) -> int:

    flags = [name for name in args if name.startswith("-")]
    file = next((name for name in args if not name.startswith("-")), "")

    if not file:
        print("No file provided.")
        return 1
    
    pass

if __name__ == "__main__":
    exit(main(sys.argv[1:]))
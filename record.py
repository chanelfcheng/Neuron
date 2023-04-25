import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", help="whether to calibrate or not", action="store_true")

    args = parser.parse_args()

    if args.calibrate:
        print("Calibrating Input...")
    else:
        print("Reading Input...")

if __name__ == "__main__":
    main()
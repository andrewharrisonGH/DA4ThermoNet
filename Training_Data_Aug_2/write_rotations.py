
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description="Write incremental rotations to a file")
    parser.add_argument("-i",  dest="increment", type=int, required=True, help="Increment in degrees around x and y axis")
    parser.add_argument("-o", dest="output_file", type=str, required=True, help="Output csv file name")
    args = parser.parse_args()

    with open(args.output_file, "w") as f:
        f.write("x_rot,y_rot,z_rot\n")
        for i in range(0, 360, args.increment):
            for j in range(0, 360, args.increment):
                f.write(f"{i},{j},0\n")


if __name__ == '__main__':
    main()
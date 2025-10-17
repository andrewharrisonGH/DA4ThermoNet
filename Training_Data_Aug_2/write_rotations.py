
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description="Write incremental rotations to a file")
    parser.add_argument("-i",  dest="increment", type=int, required=True, help="Increment in degrees around x and y axis")
    parser.add_argument("-o", dest="output_file", type=str, required=True, help="Output csv file name")
    parser.add_argument("-z", dest="z_rotation_include", action='store_true', help="Include incremental rotations around z axis too")
    args = parser.parse_args()

    with open(args.output_file, "w") as f:
        f.write("x_rot,y_rot,z_rot\n")
        for i in range(0, 360, args.increment):
            for j in range(0, 360, args.increment):
                if args.z_rotation_include:
                    for k in range(0, 360, args.increment):
                        f.write(f"{i},{j},{k}\n")
                else:
                    f.write(f"{i},{j},0\n")


if __name__ == '__main__':
    main()
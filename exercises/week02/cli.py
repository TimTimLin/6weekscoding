
from pathlib import Path

from exercises.week02.opencv import process_image

import argparse

def main()-> int:

    parser = argparse.ArgumentParser(
        description="Load, resize, and save an image."
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--height", type=int, required=True)  #代表這個參數是必須的
    parser.add_argument("--width", type=int, required=True)     
    args = parser.parse_args()
    process_image(
        input_path=args.input_path,
        output_path=args.output_path,
        height=args.height, 
        width=args.width
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
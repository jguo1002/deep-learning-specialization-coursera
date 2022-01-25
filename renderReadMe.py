"""
render latex in github markdown
https://gist.github.com/a-rodin/fef3f543412d6e1ec5b6cf55bf197d7b
"""

import argparse
import re


def generate_image(math_equation):
    print("math_equation: ", math_equation)
    render_url = 'https://render.githubusercontent.com/render/math?math='
    return f'<img src="{render_url}{math_equation}">'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fname", type=str)
    parser.add_argument("--output_fname", type=str)
    args = parser.parse_args()

    new_md = []

    with open(args.input_fname, "r") as input_f:
        text = input_f.read()

        new_md = re.sub(
            '\$[^\$]*\$', lambda x: generate_image(x.group(0)), text)

    with open(args.output_fname, "w") as output_f:
        for c in new_md:
            output_f.write(c)

# python renderReadMe.py --input_fname NOTES.md --output_fname README.md

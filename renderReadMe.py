"""
render latex in github markdown
https://gist.github.com/a-rodin/fef3f543412d6e1ec5b6cf55bf197d7b
"""

import argparse
import re


def generate_image(math_equation, color):
    math_equation = math_equation[1:-1]
    print("math_equation: ", math_equation)
    color = "\color{color}" if color != '' else ''
    render_url = f'https://render.githubusercontent.com/render/math?math={color}'
    return f'<img src="{render_url}{math_equation}">'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fname", type=str)
    parser.add_argument("--output_fname", type=str)
    args = parser.parse_args()

    new_md = []
    # default color is black so no need to specify
    outputs = [
        {
            'color': 'white',
            'fileNameSuffix': ''
        },
        {
            'color': '',
            'fileNameSuffix': 'LightMode'
        },
    ]
    with open(f"{args.input_fname}.md", "r") as input_f:
        text = input_f.read()

        for item in outputs:
            color = item['color']
            fileNameSuffix = item['fileNameSuffix']
            new_md = re.sub(
                '\$[^\$]*\$', lambda x: generate_image(x.group(0), color), text)

        with open(f"{args.output_fname}{fileNameSuffix}.md", "w") as output_f:
            for c in new_md:
                output_f.write(c)

# python renderReadMe.py --input_fname NOTES --output_fname README

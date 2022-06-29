"""
render latex in github markdown
https://gist.github.com/a-rodin/fef3f543412d6e1ec5b6cf55bf197d7b

Update: switch to codecogs because some github images returns 404
https://stackoverflow.com/questions/35498525/latex-rendering-in-readme-md-on-github
"""

import argparse
import re


def generate_image(website, math_equation, color):
    math_equation = math_equation[1:-1]
    color = "\color{" + color + "}" if color != '' else ''
    if website == "github":
        return generate_image_github(math_equation, color)
    elif website == "codecogs":
        return generate_image_codecogs(math_equation, color)


def generate_image_github(math_equation, color):
    render_url = f'https://render.githubusercontent.com/render/math?math={color}'
    return f'<img src="{render_url}{math_equation}">'


def generate_image_codecogs(math_equation, color):
    render_url = f'https://latex.codecogs.com/png.latex?{color}'
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
                '\$[^\$]*\$', lambda x: generate_image('codecogs', x.group(0), color), text)

            with open(f"{args.output_fname}{fileNameSuffix}.md", "w") as output_f:
                for c in new_md:
                    output_f.write(c)
                print(f"{args.output_fname}{fileNameSuffix} has been written")

# python renderReadMe.py --input_fname NOTES --output_fname README

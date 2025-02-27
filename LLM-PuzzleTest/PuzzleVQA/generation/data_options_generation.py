import base64
import io
import os
import itertools
import copy
import json
import math
import time
import re
import ast
import random
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fire import Fire
from pydantic import BaseModel
from tqdm import tqdm

Point = Tuple[float, float]

# def extract_list_from_caption(s):
    # match = re.search(r"\[([^\]]+)\]", s)
    # if match:
    #     # Convert the extracted string into a Python list
    #     return ast.literal_eval(f"[{match.group(1)}]")
    # return None

def extract_list_from_caption(s):
    matches = re.findall(r"\[([^\]]+)\]", s)
    list_of_lists = [ast.literal_eval(f"[{match}]") for match in matches]
    return [e for l in list_of_lists for e in l]

class CircleSizeNumberPattern(BaseModel):
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Light.ttf"
    color: str = "#eeeeee"

    def make_sample(self, caption, current_option, answer, deduction, explanation):
        # Set the size of the image
        size = self.image_size * self.scale_factor
        image = Image.new("RGB", size=(size, size), color="white")
        draw = ImageDraw.Draw(image)

        # Retrive the original list from the caption, replacing '?' with the current answer
        numbers = extract_list_from_caption(caption)
        answer_location = numbers.index('?')
        numbers = [answer if number == '?' else number for number in numbers]

        center = size // 2
        distance = 150 * self.scale_factor
        for i, number in enumerate(numbers):
            angle = (i / len(numbers)) * 2 * math.pi
            small_circle_x = center + int(distance * math.cos(angle))
            small_circle_y = center + int(distance * math.sin(angle))

            small_circle_radius = 50 * self.scale_factor + 15 * number
            draw.ellipse(
                [
                    (
                        small_circle_x - small_circle_radius,
                        small_circle_y - small_circle_radius,
                    ),
                    (
                        small_circle_x + small_circle_radius,
                        small_circle_y + small_circle_radius,
                    ),
                ],
                fill=self.color,
                outline="black",
                width=4,
            )

            draw.text(
                (
                    small_circle_x,
                    small_circle_y,
                ),
                str(number) if i != answer_location else str(current_option),
                font=ImageFont.truetype(self.path_font, size=50 * self.scale_factor),
                anchor="mm",
                fill="black",
            )

        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)

        return image


class ColorGridPattern(BaseModel):
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Medium.ttf"
    colors: Dict[str, str] = dict(
        blue="#6fa8dc",
        green="#93c47d",
        yellow="#ffd966",
        red="#e06666",
        purple="#8e7cc3",
        orange="#f6b26b",
    )

    def sample_colors(self) -> Tuple[List[str], List[str]]:
        while True:
            names = random.sample(list(self.colors), k=3)
            if "orange" in names and "yellow" in names:
                continue  # Hard to distinguish
            return names

    def draw_circle(self, draw: ImageDraw, point: Point, radius: int, color: str):
        x, y = point
        position = x - radius, y - radius, x + radius, y + radius
        line_width = self.image_size * self.scale_factor // 200
        draw.ellipse(position, fill=color, outline="black", width=line_width)

    def recover_mapping(self, grid):
        original_grid = grid[::-1]

        # Construct the mapping
        recovered_mapping = {}
        for i, label in enumerate(original_grid):
            if label not in recovered_mapping:
                recovered_mapping[label] = []
            recovered_mapping[label].append(i)

        return recovered_mapping

    def make_sample(self, caption, current_option, answer, deduction, explanation):
        # Set the size of the image
        size = self.image_size * self.scale_factor
        image = Image.new("RGB", size=(size, size), color="white")
        draw = ImageDraw.Draw(image)
        a, b, c = size // 4, size // 2, size * 3 // 4
        positions = [(x, y) for x in [a, b, c] for y in [a, b, c]]

        grid = extract_list_from_caption(caption)
        grid = [current_option if color == '?' else color for color in grid]

        mapping = self.recover_mapping(grid)

        # values = [[0, 2, 6, 8], [1, 3, 5, 7], [4]]
        # mapping = {k: v for k, v in zip(names, values)}
        # i_answer = random.choice([0, 2, 6, 8, 1, 3, 5, 7])
        # answer = ""

        for k, lst in mapping.items():
            for i in lst:
                # if i == i_answer:
                #     answer = k
                #     draw.text(
                #         positions[i],
                #         text="?",
                #         font=ImageFont.truetype(self.path_font, size=size // 10),
                #         anchor="mm",
                #         fill="black",
                #     )
                # else:
                color = self.colors[k]
                self.draw_circle(draw, positions[i], radius=size // 10, color=color)

        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        # grid = ["?"] * 9
        # for k, lst in mapping.items():
        #     for i in lst:
        #         if i != i_answer:
        #             grid[i] = k
        # grid = grid[::-1]
        # location = "at the corner" if answer == names[0] else "adjacent to the center"

        return image


class ColorHexagonPattern(BaseModel):
    colors: Dict[str, str] = dict(
        blue="#6fa8dc",
        green="#93c47d",
        yellow="#ffd966",
        red="#e06666",
        purple="#8e7cc3",
        orange="#f6b26b",
    )
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Medium.ttf"

    @staticmethod
    def get_centroid(points: List[Tuple[float, float]]) -> Tuple[float, float]:
        x = sum(p[0] for p in points) / len(points)
        y = sum(p[1] for p in points) / len(points)
        return x, y

    def sample_colors(self) -> Tuple[List[str], List[str]]:
        while True:
            names = random.sample(list(self.colors), k=3)
            if "orange" in names and "yellow" in names:
                continue  # Hard to distinguish
            names = names + names
            colors = [self.colors[n] for n in names]
            return names, colors

    def make_sample(self, caption, current_option, answer, deduction, explanation):
        # Set the size of the image
        size = self.image_size * self.scale_factor
        image = Image.new("RGB", size=(size, size), color="white")
        draw = ImageDraw.Draw(image)
        center = size // 2

        # Hexagon properties
        side_length = size // 3  # Length of a side of the hexagon and triangles
        triangle_height = math.sqrt(3) / 2 * side_length

        # The vertices of the hexagon
        hexagon = [
            (center + side_length / 2, center - triangle_height),
            (center - side_length / 2, center - triangle_height),
            (center - side_length, center),
            (center - side_length / 2, center + triangle_height),
            (center + side_length / 2, center + triangle_height),
            (center + side_length, center),
        ]

        # Colors for the triangles
        names = extract_list_from_caption(caption)
        names = [current_option if color == '?' else color for color in names]
        colors = [self.colors[n] for n in names]

        # Draw the hexagon made of six triangles
        for i in range(6):
            # Coordinates of the triangle vertices
            triangle = [hexagon[i], hexagon[(i + 1) % 6], (center, center)]
            # Draw the triangle
            draw.polygon(triangle, fill=colors[i])
            # Draw the outline with custom width
            points = [hexagon[i], hexagon[(i + 1) % 6], (center, center), hexagon[i]]
            draw.line(points, fill="black", width=self.scale_factor * 4)

        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        return image


class ColorNumberHexagonPattern(BaseModel):
    colors: Dict[str, str] = dict(
        blue="#9ec5e8",
        green="#b6d7a8",
        yellow="#fee599",
        red="#ea9999",
        purple="#b4a7d6",
        orange="#f9cb9c",
    )
    numbers: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Medium.ttf"

    @staticmethod
    def get_centroid(points: List[Tuple[float, float]]) -> Tuple[float, float]:
        x = sum(p[0] for p in points) / len(points)
        y = sum(p[1] for p in points) / len(points)
        return x, y

    def sample_colors(self) -> Tuple[List[str], List[str]]:
        while True:
            names = random.sample(list(self.colors), k=3)
            if "orange" in names and "yellow" in names:
                continue  # Hard to distinguish
            names = names + names
            colors = [self.colors[n] for n in names]
            return names, colors

    def make_sample(self, caption, current_option, answer, deduction, explanation):
        # Set the size of the image
        size = self.image_size * self.scale_factor
        image = Image.new("RGB", size=(size, size), color="white")
        draw = ImageDraw.Draw(image)
        center = size // 2

        # Hexagon properties
        side_length = size // 3  # Length of a side of the hexagon and triangles
        triangle_height = math.sqrt(3) / 2 * side_length

        # The vertices of the hexagon
        hexagon = [
            (center + side_length / 2, center - triangle_height),
            (center - side_length / 2, center - triangle_height),
            (center - side_length, center),
            (center - side_length / 2, center + triangle_height),
            (center + side_length / 2, center + triangle_height),
            (center + side_length, center),
        ]

        # Colors for the triangles
        names = extract_list_from_caption(caption)
        names = [current_option if color == '?' else color for color in names]
        numbers = names[6:]
        names = names[:6]

        colors = [self.colors[n] for n in names]

        # Draw the hexagon made of six triangles
        for i in range(6):
            # Coordinates of the triangle vertices
            triangle = [hexagon[i], hexagon[(i + 1) % 6], (center, center)]
            # Draw the triangle
            draw.polygon(triangle, fill=colors[i])
            # Draw the outline with custom width
            points = [hexagon[i], hexagon[(i + 1) % 6], (center, center), hexagon[i]]
            draw.line(points, fill="black", width=self.scale_factor * 4)
            # Add number or "?" for missing part
            draw.text(
                self.get_centroid(triangle),
                text=str(numbers[i]),
                font=ImageFont.truetype(self.path_font, size=size // 10),
                anchor="mm",
                fill="black",
            )

        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)

        return image


class ColorOverlapSquaresPattern(BaseModel):
    colors: Dict[str, str] = dict(
        blue="#6fa8dc",
        green="#93c47d",
        yellow="#ffd966",
        red="#e06666",
        purple="#8e7cc3",
        orange="#f6b26b",
    )
    numbers: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Medium.ttf"
    num_sides: int = 4

    def get_points(self, center: Point, radius: float, angle: int = 0) -> List[Point]:
        vertices = []
        for i in range(self.num_sides):
            theta = 2 * math.pi / self.num_sides * i
            theta -= math.pi / 4  # Adjust to flat rectangle by default
            theta -= math.radians(angle)
            x = center[0] + radius * math.cos(theta)
            y = center[1] + radius * math.sin(theta)
            vertices.append((x, y))
        return vertices

    def draw_squares(self, draw: ImageDraw, colors: List[str]):
        size = self.image_size * self.scale_factor
        line_width = size // 150

        # Center big square
        a = self.get_points((size / 2, size / 2), radius=size / 4)
        draw.polygon(a, outline="black", fill=colors[1], width=line_width)

        # Top right rotated square
        b = self.get_points(a[0], radius=size / 4, angle=45)
        draw.polygon(b, outline="black", fill=colors[2], width=line_width)

        # Bottom left rotated square
        c = self.get_points(a[2], radius=size / 4, angle=45)
        draw.polygon(c, outline="black", fill=colors[0], width=line_width)

        # Top right overlap triangle
        ab = [a[0], b[2], b[3]]
        draw.polygon(ab, outline="black", fill=colors[4], width=line_width)

        # Bottom left overlap triangle
        ac = [a[2], c[0], c[1]]
        draw.polygon(ac, outline="black", fill=colors[3], width=line_width)

    def sample_color_names(self) -> Tuple[str, str, str, str, str]:
        a, b, c = random.sample(["red", "yellow", "blue"], k=3)
        mapping = dict(redyellow="orange", blueyellow="green", bluered="purple")
        d = mapping["".join(sorted([a, b]))]
        e = mapping["".join(sorted([b, c]))]
        assert [x in self.colors for x in [a, b, c, d, e]]
        return a, b, c, d, e

    def make_sample(self, caption, current_option, answer, deduction, explanation):
        # Set the size of the image
        size = self.image_size * self.scale_factor
        image = Image.new("RGB", size=(size, size), color="white")
        draw = ImageDraw.Draw(image)

        color_names = extract_list_from_caption(caption)
        color_names = [current_option if color == '?' else color for color in color_names]
        color_names += caption.replace('.', '').split('and second squares overlap is ')[1].split(' The part where the second and third squares overlap is ')

        colors = [self.colors[n] for n in color_names]

        self.draw_squares(draw, colors)

        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)

        return image


class ColorSizeCirclePattern(BaseModel):
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Medium.ttf"
    colors: Dict[str, str] = dict(
        blue=["#cfe2f3", "#9fc4e8", "#6fa7dc", "#3d85c5"],
        green=["#d9ead3", "#b6d7a8", "#92c47d", "#69a84f"],
        yellow=["#fff2cc", "#fee599", "#ffd966", "#f0c232"],
        red=["#f4cccc", "#ea9999", "#e06666", "#cc0101"],
        purple=["#d9d2e9", "#b3a7d6", "#8d7cc3", "#664ea6"],
        orange=["#fbe5cd", "#f9ca9c", "#f6b16b", "#e69139"],
    )

    def draw_circle(self, draw: ImageDraw, x: int, y: int, radius: int, **kwargs):
        position = x - radius, y - radius, x + radius, y + radius
        line_width = self.image_size * self.scale_factor // 150
        draw.ellipse(position, width=line_width, **kwargs)

    def make_sample(self, caption, current_option, answer, deduction, explanation):
        # Set the size of the image
        size = self.image_size * self.scale_factor
        image = Image.new("RGB", size=(size, size), color="white")
        draw = ImageDraw.Draw(image)

        keys = extract_list_from_caption(caption)
        keys = [k for k in keys[-4:] if k!='?']
        key = keys[0].split()[-1]

        colors = deepcopy(self.colors[key])

        if current_option != answer:
            dark_light, color = current_option.split()
            colors[-1] = self.colors[color][0] if dark_light=='light' else self.colors[color][-1]

        radii = [size * 0.4, size * 0.3, size * 0.2, size * 0.1]
        for i, r in enumerate(radii):
            x = y = size // 2
            fill = colors[i]
            self.draw_circle(draw, x, y, r, fill=fill, outline="black")

        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        return image


class GridNumberColorPattern(BaseModel):
    colors: Dict[str, str] = dict(
        blue="#6fa8dc",
        green="#93c47d",
        yellow="#ffd966",
        red="#e06666",
        purple="#8e7cc3",
        orange="#f6b26b",
    )
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Light.ttf"

    def make_sample(self, caption, current_option, answer, deduction, explanation):
        size = self.image_size * self.scale_factor
        buffer = 30 * self.scale_factor
        num_rows = 3
        num_cols = 3

        image = Image.new("RGB", (size + buffer * 2, size + buffer * 2), "white")

        # Get the drawing context
        draw = ImageDraw.Draw(image)
        length = size // num_cols
        height = size // num_rows

        # Randomly choose 4 integers that will appear in the matrix
        matrix = extract_list_from_caption(caption)
        matrix = [(n ,c) if c != '?' else (n, current_option) for (n, c) in matrix]
        matrix = [matrix[0:3], matrix[3:6], matrix[6:]]

        # Draw shapes and numbers
        for i, row in enumerate(matrix):
            for j, (num, col) in enumerate(row):
                color = (
                    self.colors[col]
                )
                number = str(num)

                draw.rounded_rectangle(
                    (
                        buffer + j * length,
                        buffer + i * height,
                        buffer + (j + 1) * length,
                        buffer + (i + 1) * height,
                    ),
                    fill=color,
                    outline="black",
                    width=4,
                    radius=size // 20,
                )

                draw.text(
                    (
                        buffer + (j * length) + (length // 2),
                        buffer + (i * height) + (height // 2),
                    ),
                    number,
                    font=ImageFont.truetype(
                        self.path_font, size=60 * self.scale_factor
                    ),
                    anchor="mm",
                    fill="black",
                )

        return image


class GridNumberPattern(BaseModel):
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Medium.ttf"
    color: str = "#cfe2f3"  # Light blue

    def draw_text(self, draw: ImageDraw, point: Point, text: str):
        size = self.image_size * self.scale_factor
        draw.text(
            point,
            text=text,
            font=ImageFont.truetype(self.path_font, size=size // 8),
            anchor="mm",
            fill="black",
        )

    def draw_box(self, draw: ImageDraw, point: Point):
        size = self.image_size * self.scale_factor
        width = height = size / 8
        draw.rounded_rectangle(
            [point[0] - width, point[1] - height, point[0] + width, point[1] + height],
            outline="black",
            width=size // 200,
            radius=size // 20,
            fill=self.color,
        )

    def make_sample(self, caption, current_option, answer, deduction, explanation):
        size = self.image_size * self.scale_factor
        image = Image.new("RGB", (size, size), "white")
        draw = ImageDraw.Draw(image)
        # values = random.sample(range(1, 10), 6)
        # num_rows = 3

        # Adjust the matrix to ensure that the sum of each row and column is the same
        matrix = extract_list_from_caption(caption)
        matrix = [n if n != '?' else current_option for n in matrix]
        matrix = [matrix[:3], matrix[3:6], matrix[6:9]]

        # while True:
        #     matrix = np.random.choice(values, size=(3, 3))
        #     row_sums = matrix.sum(axis=1)
        #     desired_sum = row_sums[0]

        #     if np.all(row_sums == desired_sum) and len(set(matrix.flatten())) >= 4:
        #         break

        # answer_location = np.random.randint(0, num_rows, size=2)
        # answer = matrix[answer_location[0]][answer_location[1]]
        # matrix = matrix.tolist()
        # matrix[answer_location[0]][answer_location[1]] = "?"

        a, b, c = size * 0.25, size * 0.50, size * 0.75
        locations = [
            [(a, a), (b, a), (c, a)],
            [(a, b), (b, b), (c, b)],
            [(a, c), (b, c), (c, c)],
        ]

        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                self.draw_box(draw, point=locations[i][j])
                self.draw_text(draw, point=locations[i][j], text=str(val))

        # values.remove(answer)
        # values = values[:3]
        # values.append(int(answer))
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)

        # instances = [row for row in matrix if "?" not in row]
        # instances.extend([row for row in matrix if "?" in row])
        # assert len(instances) == len(matrix)

        return image


class PolygonSidesColorPattern(BaseModel):
    colors: Dict[str, str] = dict(
        blue="#6fa8dc",
        green="#93c47d",
        yellow="#ffd966",
        red="#e06666",
        purple="#8e7cc3",
        orange="#f6b26b",
    )
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Light.ttf"

    @staticmethod
    def draw_polygon(draw, sides, center, size, color):
        angle = 360 / sides
        points = []

        for i in range(sides):
            x = center[0] + size * math.cos(math.radians(i * angle))
            y = center[1] + size * math.sin(math.radians(i * angle))
            points.append((x, y))

        draw.polygon(points, outline="black", fill=color, width=4)

    def make_sample(self, caption, current_option, answer, deduction, explanation):
        # Set the size of the image
        size = self.image_size * self.scale_factor
        image = Image.new("RGB", size=(size, size), color="white")
        draw = ImageDraw.Draw(image)

        colors = extract_list_from_caption(caption)
        colors = [n if n != '?' else current_option for n in colors]

        colors = [colors[0], colors[1], colors[3], colors[4], colors[5], colors[2]]

        col2side = {}
        for expl in explanation.split('the polygon with ')[1:]:
            s, c = expl.split(' sides is ')
            c = c.split()[0]
            s = int(s)
            col2side[c] = s
        
        s, c = deduction.replace('Based on the pattern that the polygons with the same number of sides have the same color, the missing color of the part with ', '').replace('.', '').split(' sides should be ')
        s = int(s)
        col2side[c] = s

        if current_option not in col2side:
            col2side[current_option] = col2side[answer]

        center = size // 2
        distance = 175 * self.scale_factor

        for i, color in enumerate(colors):
            polygon_distance = distance - 0.5 * (i % 2) * distance
            angle = (i / len(colors)) * 2 * math.pi
            center_y = center - int(polygon_distance * math.cos(angle))
            center_x = center - int(polygon_distance * math.sin(angle))
            polygon_size = 60 * self.scale_factor

            self.draw_polygon(draw, col2side[color], (center_x, center_y), polygon_size, self.colors[color])

        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)

        return image


class PolygonSidesNumberPattern(BaseModel):
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Medium.ttf"
    color: str = "#d9d2e8"  # Light purple

    def draw_polygon(self, draw, sides, center, size):
        angle = 360 / sides
        points = []

        for i in range(sides):
            x = center[0] + size * math.cos(math.radians(i * angle))
            y = center[1] + size * math.sin(math.radians(i * angle))
            points.append((x, y))

        draw.polygon(points, outline="black", fill=self.color, width=12)

    def make_sample(self, caption, current_option, answer, deduction, explanation):
        # Set the size of the image
        size = self.image_size * self.scale_factor
        image = Image.new("RGB", size=(size, size), color="white")
        draw = ImageDraw.Draw(image)

        sides = extract_list_from_caption(caption)
        sides = [sides[0], sides[1], sides[3], sides[4], sides[5], sides[2]]
        answer_location = sides.index('?')
        sides = [n if n != '?' else answer for n in sides]

        center = size // 2
        distance = 175 * self.scale_factor

        for i, side in enumerate(sides):
            polygon_distance = distance - 0.5 * (i % 2) * distance
            angle = (i / len(sides)) * 2 * math.pi
            center_y = center - int(polygon_distance * math.cos(angle))
            center_x = center - int(polygon_distance * math.sin(angle))
            polygon_size = 60 * self.scale_factor

            self.draw_polygon(draw, side, (center_x, center_y), polygon_size)

            # Draw text in the center of the polygon
            draw.text(
                (center_x, center_y),
                str(side) if i != answer_location else str(current_option),
                font=ImageFont.truetype(self.path_font, size=50 * self.scale_factor),
                anchor="mm",
                fill="black",
            )

        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)

        return image


class RectangleHeightColorPattern(BaseModel):
    colors: Dict[str, str] = dict(
        blue="#6fa8dc",
        green="#93c47d",
        yellow="#ffd966",
        red="#e06666",
        purple="#8e7cc3",
        orange="#f6b26b",
    )
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Light.ttf"

    def draw_box(
        self,
        draw: ImageDraw,
        point: Point,
        width: float,
        height: float,
        color: str,
    ):
        size = self.image_size * self.scale_factor
        draw.rounded_rectangle(
            [point[0] - width, point[1] - height, point[0] + width, point[1] + height],
            outline="black",
            width=size // 200,
            radius=size // 20,
            fill=color,
        )

    def draw_text(self, draw: ImageDraw, point: Point, text: str):
        size = self.image_size * self.scale_factor
        draw.text(
            point,
            text=text,
            font=ImageFont.truetype(self.path_font, size=size // 8),
            anchor="mm",
            fill="black",
        )

    @staticmethod
    def assign_numbers(colors: List[str]) -> List[int]:
        unique = sorted(set(colors))
        numbers = [i + 1 for i in range(len(unique))]
        random.shuffle(numbers)
        mapping = {u: i for u, i in zip(unique, numbers)}
        return [mapping[c] for c in colors]

    def make_sample(self, caption, current_option, answer, deduction, explanation):
        size = self.image_size * self.scale_factor
        image = Image.new("RGB", (size, size), "white")
        draw = ImageDraw.Draw(image)

        colors = extract_list_from_caption(caption)
        lengths = colors[:7]
        colors = colors[7:]

        mapping = {'short': 1, 'medium': 2, 'long': 3}
        numbers = [mapping[l] for l in lengths]

        for i, num in enumerate(numbers):
            factor = size / (len(numbers) + 1)
            point = (factor * (i + 1), size // 2)
            is_answer = colors[i] == '?'
            self.draw_box(
                draw,
                point=point,
                width=factor / 2,
                height=factor * num,
                color=self.colors[current_option] if is_answer else self.colors[colors[i]],
            )

        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)

        return image


class RectangleHeightNumberPattern(BaseModel):
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Medium.ttf"
    color: str = "#d9ead3"  # Light green

    def draw_text(self, draw: ImageDraw, point: Point, text: str):
        size = self.image_size * self.scale_factor
        draw.text(
            point,
            text=text,
            font=ImageFont.truetype(self.path_font, size=size // 10),
            anchor="mm",
            fill="black",
        )

    def draw_box(self, draw: ImageDraw, point: Point, width: float, height: float):
        size = self.image_size * self.scale_factor
        draw.rounded_rectangle(
            [point[0] - width, point[1] - height, point[0] + width, point[1] + height],
            outline="black",
            width=size // 200,
            radius=size // 20,
            fill=self.color,
        )

    def make_sample(self, caption, current_option, answer, deduction, explanation):
        size = self.image_size * self.scale_factor
        image = Image.new("RGB", (size, size), "white")
        draw = ImageDraw.Draw(image)

        numbers = extract_list_from_caption(caption)
        numbers = numbers[:7]
        answer_i = numbers.index('?')
        numbers = [n if n != '?' else answer for n in numbers]

        for i, num in enumerate(numbers):
            factor = size / (len(numbers) + 1)
            point = (factor * (i + 1), size // 2)
            self.draw_box(
                draw,
                point=point,
                width=factor / 2,
                height=factor * int(num),
            )
            self.draw_text(draw, point, str(current_option) if i == answer_i else str(num))

        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        lengths = [["short", "medium", "long"][num - 1] for num in numbers]
        numbers[-1] = "?"

        return image


def get_polygon_point(num_sides: int, r: int, angle: int) -> Tuple[float, float]:
    # Find any point on a regular polygon as a function of angle from center (0 to 360)
    theta = math.radians(angle) % (2 * math.pi)  # Normalize to within 0 to 2π
    if num_sides == 4:
        theta += math.pi // 4
    alpha = 2 * math.pi / num_sides  # Angle per segment in radians
    vertex_before = int(theta // alpha)  # Nearest vertex before the point
    theta_vertex = vertex_before * alpha  # Angle to the nearest vertex before the point
    if theta == theta_vertex:
        return r * math.cos(theta), r * math.sin(theta)

    # Find the coordinates of the two nearest vertices
    x1, y1 = r * math.cos(theta_vertex), r * math.sin(theta_vertex)
    x2, y2 = r * math.cos(theta_vertex + alpha), r * math.sin(theta_vertex + alpha)

    # Linear interpolation on the edge
    t = (theta - theta_vertex) / alpha
    x = (1 - t) * x1 + t * x2
    y = (1 - t) * y1 + t * y2
    return x, y


class ShapeMorphPattern(BaseModel):
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Medium.ttf"
    color: str = "#f4cccb"  # Light red for all shapes
    shapes: Dict[str, int] = dict(
        triangle=3, square=4, pentagon=5, hexagon=6, circle=720
    )

    @staticmethod
    def interpolate_points(
        points_a: List[Tuple[float, float]],
        points_b: List[Tuple[float, float]],
        weight: float,
    ) -> List[Tuple[float, float]]:
        outputs = []
        assert len(points_a) == len(points_b)
        assert 0 <= weight <= 1
        for a, b in zip(points_a, points_b):
            x = a[0] * (1 - weight) + b[0] * weight
            y = a[1] * (1 - weight) + b[1] * weight
            outputs.append((x, y))
        return outputs

    @staticmethod
    def offset_points(
        points: List[Tuple[float, float]], x: float, y: float
    ) -> List[Tuple[float, float]]:
        return [(p[0] + x, p[1] + y) for p in points]

    def draw_text(self, draw: ImageDraw, x: float, y: float, text: str):
        size = self.image_size * self.scale_factor
        draw.text(
            (x, y),
            text=text,
            font=ImageFont.truetype(self.path_font, size=size // 8),
            anchor="mm",
            fill="black",
        )

    def make_sample(self, caption, current_option, answer, deduction, explanation):
        # Set the size of the image
        size = self.image_size * self.scale_factor
        image = Image.new("RGB", size=(size, size), color="white")
        draw = ImageDraw.Draw(image)

        names = caption.replace('There are eight shapes arranged in a grid. The top left shape is a ', '') \
               .replace('. The other shapes do not appear to regular shapes.', '') \
               .split(' and the bottom right shape is a ')
        
        answer_i = names.index('?')
        names = [n if n != '?' else answer for n in names]

        radius = size // 10
        angles = list(range(360))
        points_a = [get_polygon_point(self.shapes[names[0]], radius, i) for i in angles]
        points_e = [get_polygon_point(self.shapes[names[1]], radius, i) for i in angles]
        points_b = self.interpolate_points(points_a, points_e, weight=0.25)
        points_c = self.interpolate_points(points_a, points_e, weight=0.50)
        points_d = self.interpolate_points(points_a, points_e, weight=0.75)

        if answer_i == 0:
            points_a = [get_polygon_point(self.shapes[current_option], radius, i) for i in angles]
        else:
            points_e = [get_polygon_point(self.shapes[current_option], radius, i) for i in angles]

        for lst, (x, y) in [
            (points_a, (size * 1 // 4, size * 1 // 4)),
            (points_b, (size * 2 // 4, size * 1 // 4)),
            (points_c, (size * 3 // 4, size * 1 // 4)),
            (points_d, (size * 3 // 4, size * 2 // 4)),
            # (points_c, (size * 2 // 4, size * 2 // 4)),
            (points_b, (size * 1 // 4, size * 2 // 4)),
            (points_c, (size * 1 // 4, size * 3 // 4)),
            (points_d, (size * 2 // 4, size * 3 // 4)),
            (points_e, (size * 3 // 4, size * 3 // 4)),
        ]:
            lst = self.offset_points(lst, x=x, y=y)
            draw.polygon(lst, fill=self.color, outline="black", width=size // 150)

        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)

        return image


class ShapeReflectPattern(BaseModel):
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Medium.ttf"
    color: str = "#d9ead3"  # Light green for all shapes
    shapes: Dict[str, int] = dict(triangle=3, square=4, pentagon=5, hexagon=6)

    def draw_circle(self, draw: ImageDraw, x: int, y: int, radius: int, **kwargs):
        position = x - radius, y - radius, x + radius, y + radius
        line_width = self.image_size * self.scale_factor // 200
        draw.ellipse(position, width=line_width, **kwargs)

    def draw_dotted_circle(
        self, draw: ImageDraw, center: Tuple[float, float], radius: int, num_dots: int
    ):
        self.draw_circle(draw, *center, radius, outline="black")
        angle_between_dots = 2 * math.pi / num_dots
        for i in range(0, num_dots, 2):
            theta = angle_between_dots * i
            x = round(center[0] + radius * math.cos(theta))
            y = round(center[1] + radius * math.sin(theta))
            self.draw_circle(draw, x, y, radius=radius // 10, fill="white")

    def draw_shape(
        self,
        draw: ImageDraw,
        center: Tuple[float, float],
        num_sides: int,
        do_flip: bool,
    ):
        size = self.image_size * self.scale_factor
        if num_sides == 0:
            draw.text(
                center,
                text="?",
                font=ImageFont.truetype(self.path_font, size=size // 10),
                anchor="mm",
                fill="black",
            )
            self.draw_dotted_circle(draw, center, radius=size // 10, num_dots=32)
            return

        # Adjust start angle based on even or odd number of sides
        angle = math.pi * 2 / num_sides
        if num_sides % 2 == 0:
            start = math.pi / 2 - angle / 2
        else:
            start = 0
        if do_flip:
            start += math.pi

        radius = size // 10
        points = [
            (
                center[0] + math.sin(start + angle * i) * radius,
                center[1] - math.cos(start + angle * i) * radius,
            )
            for i in range(num_sides)
        ]

        width = size // 200
        draw.polygon(points, fill=self.color, outline="black", width=width)
        draw.line([(size // 8, size // 2), (size * 7 // 8, size // 2)], "black", width)

    def make_sample(self, caption, current_option, answer, deduction, explanation):
        # Set the size of the image
        size = self.image_size * self.scale_factor
        image = Image.new("RGB", size=(size, size), color="white")
        draw = ImageDraw.Draw(image)

        names = extract_list_from_caption(caption)
        names = [n if n != '?' else current_option for n in names]

        a, b, c = size // 4, size // 2, size * 3 // 4
        positions = [(x, y) for y in [size // 3, size * 2 // 3] for x in [a, b, c]]

        for i, n in enumerate(names):
            num_sides = self.shapes[n]
            self.draw_shape(draw, positions[i], num_sides, do_flip=i >= 3)

        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        return image


class ShapeSizeGridPattern(BaseModel):
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Medium.ttf"
    color: str = "#d9ead3"  # Light green for all shapes
    shapes: Dict[str, int] = dict(triangle=3, square=4, pentagon=5, hexagon=6)

    @staticmethod
    def get_points(num_sides: int, center: Point, radius: int) -> List[Point]:
        vertices = []
        for i in range(num_sides):
            theta = 2 * math.pi / num_sides * i
            if num_sides % 2 != 0:
                theta -= math.pi / 2
            elif num_sides == 4:
                theta -= math.pi / 4

            x = center[0] + radius * math.cos(theta)
            y = center[1] + radius * math.sin(theta)
            vertices.append((x, y))
        return vertices

    def draw_text(self, draw: ImageDraw, point: Point, text: str):
        size = self.image_size * self.scale_factor
        draw.text(
            point,
            text=text,
            font=ImageFont.truetype(self.path_font, size=size // 8),
            anchor="mm",
            fill="black",
        )

    @staticmethod
    def random_rotate_matrix(matrix: List[list]) -> List[list]:
        angle = random.choice([90, 180, 270, 360])
        if angle == 90:
            # Rotate by 90 degrees
            new = [list(row) for row in zip(*matrix[::-1])]
        elif angle == 180:
            # Rotate by 180 degrees
            new = [row[::-1] for row in matrix[::-1]]
        elif angle == 270:
            # Rotate by 270 degrees (or 90 degrees counter-clockwise)
            new = [list(row) for row in zip(*matrix)][::-1]
        else:
            new = matrix

        return [
            [
                (new[i][j][0], new[i][j][1], matrix[i][j][2], matrix[i][j][3])
                for j in range(len(matrix[i]))
            ]
            for i in range(len(matrix))
        ]

    def make_sample(self, caption, current_option, answer, deduction, explanation):
        size = self.image_size * self.scale_factor
        image = Image.new("RGB", size=(size, size), color="white")
        draw = ImageDraw.Draw(image)

        answer_shape = deduction.split(', the size of the missing ')[1].split()[0]
        data = extract_list_from_caption(caption)
        data = [d if d != '?' else f'{current_option} {answer_shape}' for d in data]

        mapping = dict(small=(size * 0.05), medium=(size * 0.09), large=(size * 0.13))
        d, e, f = size * 0.25, size * 0.50, size * 0.75
        data = [
            [(data[0].split()[1], data[0].split()[0], d, d), (data[1].split()[1], data[1].split()[0], e, d), (data[2].split()[1], data[2].split()[0], f, d)],
            [(data[3].split()[1], data[3].split()[0], d, e), (data[4].split()[1], data[4].split()[0], e, e), (data[5].split()[1], data[5].split()[0], f, e)],
            [(data[6].split()[1], data[6].split()[0], d, f), (data[7].split()[1], data[7].split()[0], e, f), (data[8].split()[1], data[8].split()[0], f, f)]
        ]
        for lst in data:
            for item in lst:
                name, radius, x, y = item
                if item == answer:
                    self.draw_text(draw, point=(x, y), text="?")
                    continue
                shape = self.get_points(
                    num_sides=self.shapes[name],
                    center=(x, y),
                    radius=mapping[radius],
                )
                draw.polygon(shape, fill=self.color, outline="black", width=size // 100)

        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)

        return image


class ShapeSizeHexagonPattern(BaseModel):
    image_size: int = 512
    scale_factor: int = 4
    path_font: str = "fonts/OpenSans-Medium.ttf"
    color: str = "#fce5cd"  # Light orange for all shapes
    shapes: Dict[str, int] = dict(triangle=3, square=4, pentagon=5, hexagon=6)

    @staticmethod
    def get_points(num_sides: int, center: Point, radius: int) -> List[Point]:
        vertices = []
        for i in range(num_sides):
            theta = 2 * math.pi / num_sides * i
            if num_sides % 2 != 0:
                theta -= math.pi / 2
            elif num_sides == 4:
                theta -= math.pi / 4

            x = center[0] + radius * math.cos(theta)
            y = center[1] + radius * math.sin(theta)
            vertices.append((x, y))
        return vertices

    def draw_text(self, draw: ImageDraw, point: Point, text: str):
        size = self.image_size * self.scale_factor
        draw.text(
            point,
            text=text,
            font=ImageFont.truetype(self.path_font, size=size // 6),
            anchor="mm",
            fill="black",
        )

    def make_sample(self, caption, current_option, answer, deduction, explanation):
        size = self.image_size * self.scale_factor
        image = Image.new("RGB", size=(size, size), color="white")
        draw = ImageDraw.Draw(image)
        center = size // 2, size // 2

        mapping = dict(small=(size * 0.05), medium=(size * 0.10), large=(size * 0.15))

        answer_shape = deduction.split('Based on the pattern that each shape appears with a distinct size, the size of the missing ')[1].split()[0]

        shape_names = extract_list_from_caption(caption)
        size_names = shape_names[9:]
        shape_names = shape_names[:3]

        indices = [0, 1, 2, 0, 1, 2]
        points = self.get_points(num_sides=6, center=center, radius=size // 3)

        for i, p in zip(indices, points):
            shape = self.get_points(
                num_sides=self.shapes[shape_names[i]],
                center=p,
                radius=mapping[size_names[i]],
            )
            draw.polygon(shape, fill=self.color, outline="black", width=size // 100)

        answer_size = deepcopy(current_option)
        shape = self.get_points(
            num_sides=self.shapes[answer_shape],
            center=center,
            radius=mapping[answer_size],
        )
        draw.polygon(shape, fill=self.color, outline="black", width=size // 100)

        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        return image


# class SizeCyclePattern(BaseModel):
#     image_size: int = 512
#     scale_factor: int = 4
#     path_font: str = "fonts/OpenSans-Medium.ttf"
#     color: str = "#fff2cc"  # Light yellow for all circles

#     def draw_circle(self, draw: ImageDraw, point: Point, radius: int):
#         x, y = point
#         position = x - radius, y - radius, x + radius, y + radius
#         line_width = self.image_size * self.scale_factor // 150
#         draw.ellipse(position, fill=self.color, outline="black", width=line_width)

#     @staticmethod
#     def get_points(n_sides: int, center: Point, radius: int, angle: int) -> List[Point]:
#         def regular_polygon_vertices(num_sides):
#             vertices = []
#             for i in range(num_sides):
#                 theta = 2 * math.pi / num_sides * i
#                 x = center[0] + radius * math.cos(theta)
#                 y = center[1] + radius * math.sin(theta)
#                 vertices.append((x, y))
#             return vertices

#         def rotate_point(origin, point):
#             ox, oy = origin
#             px, py = point
#             theta = math.radians(angle)  # Convert to radians
#             qx = ox + math.cos(theta) * (px - ox) - math.sin(theta) * (py - oy)
#             qy = oy + math.sin(theta) * (px - ox) + math.cos(theta) * (py - oy)
#             return qx, qy

#         polygon_vertices = regular_polygon_vertices(n_sides)
#         # assert self.get_centroid(polygon_vertices) == center
#         rotated_vertices = [rotate_point(center, v) for v in polygon_vertices]
#         # assert self.get_centroid(rotated_vertices) == center
#         return rotated_vertices

#     def draw_text(self, draw: ImageDraw, point: Point, text: str):
#         size = self.image_size * self.scale_factor
#         draw.text(
#             point,
#             text=text,
#             font=ImageFont.truetype(self.path_font, size=size // 10),
#             anchor="mm",
#             fill="black",
#         )

#     def make_sample(self, caption, current_option, answer, deduction, explanation):
#         # Set the size of the image
#         size = self.image_size * self.scale_factor
#         image = Image.new("RGB", size=(size, size), color="white")
#         draw = ImageDraw.Draw(image)

#         center = size // 2, size // 2
#         offset = random.randint(0, 360)
#         mapping = dict(
#             small=(size * 0.050, size // 9, 0 + offset),
#             medium=(size * 0.075, size // 4, 20 + offset),
#             large=(size * 0.100, size // 2.5, 45 + offset),
#         )

#         names = []
#         num_sides = 3
#         answer = ""
#         i_answer = random.randint(0, num_sides * len(mapping) - 1)
#         for n, (radius, distance, angle) in mapping.items():
#             for p in self.get_points(num_sides, center, distance, angle):
#                 names.append(n)
#                 if len(names) - 1 == i_answer:
#                     self.draw_text(draw, p, "?")
#                     answer = n
#                 else:
#                     self.draw_circle(draw, p, round(radius))

#         names[i_answer] = "?"
#         arms = [
#             [names[0], names[3], names[6]],
#             [names[1], names[4], names[7]],
#             [names[2], names[5], names[8]],
#         ]
#         image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
#         answer_location = dict(
#             small="closest to center",
#             medium="neither closest nor farthest from center",
#             large="farthest from center",
#         )[answer]

#         return (
#             dict(
#                 question="What is the size of the missing circle denoted with a question mark?",
#                 answer=answer,
#                 options=sample_options(answer, sorted(mapping), k=3),
#                 caption=f"There are circles arranged in a spiral with three arms. The first arm has circles of sizes {arms[0]}, the second arm has circles of sizes {arms[1]}, and the third arm has circles of sizes {arms[2]}.",
#                 explanation=f"We observe that the circles in each arm progress in size from small to medium to large. Thus, the pattern is that the circles in each arm get bigger as they progress away from the center of the spiral.",
#                 deduction=f"Based on the pattern that the circles in each arm get bigger as they progress away from the center of the spiral, the size of the missing part that is {answer_location} should be {answer}.",
#             ),
#             image,
#         )


# class SizeGridPattern(BaseModel):
#     image_size: int = 512
#     scale_factor: int = 4
#     path_font: str = "fonts/OpenSans-Medium.ttf"
#     color: str = "#fff2cc"  # Light yellow for all circles

#     def draw_circle(self, draw: ImageDraw, x: int, y: int, radius: int):
#         position = x - radius, y - radius, x + radius, y + radius
#         line_width = self.image_size * self.scale_factor // 200
#         draw.ellipse(position, fill=self.color, outline="black", width=line_width)

#     def make_sample(self, caption, current_option, answer, deduction, explanation):
#         # Set the size of the image
#         size = self.image_size * self.scale_factor
#         image = Image.new("RGB", size=(size, size), color="white")
#         draw = ImageDraw.Draw(image)
#         a, b, c = size // 4, size // 2, size * 3 // 4
#         positions = [(x, y) for x in [a, b, c] for y in [a, b, c]]

#         radii = dict(small=size // 30, medium=size // 20, large=size // 10)
#         keys = random.sample(radii.keys(), k=len(radii))
#         values = [[0, 2, 6, 8], [1, 3, 5, 7], [4]]
#         mapping = {k: v for k, v in zip(keys, values)}
#         i_answer = random.choice([0, 2, 6, 8, 1, 3, 5, 7])
#         answer = ""

#         for k, lst in mapping.items():
#             radius = radii[k]
#             for i in lst:
#                 if i == i_answer:
#                     answer = k
#                     draw.text(
#                         positions[i],
#                         text="?",
#                         font=ImageFont.truetype(self.path_font, size=size // 10),
#                         anchor="mm",
#                         fill="black",
#                     )
#                 else:
#                     self.draw_circle(draw, *positions[i], radius=radius)

#         options = sample_options(answer, list(radii.keys()), k=3)
#         image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
#         grid = ["?"] * 9
#         for k, lst in mapping.items():
#             for i in lst:
#                 if i != i_answer:
#                     grid[i] = k
#         grid = grid[::-1]
#         answer_location = (
#             "at the corner" if i_answer in [0, 2, 6, 8] else "adjacent to the center"
#         )

#         return (
#             dict(
#                 question="What is the size of the missing part denoted with a question mark?",
#                 answer=answer,
#                 options=options,
#                 caption=f"There are circles arranged in a grid formation with varying sizes in the image. The sizes in the first row are {grid[:3]}, the sizes in the second row are {grid[3:6]}, and the sizes in the third row are {grid[6:9]}.",
#                 explanation=f"We observe that the circles at the corners are {keys[0]} size, while the circles directly adjacent to the center are {keys[1]} size. Only the center circle is {keys[2]} size. Hence, the pattern is that the circles alternate in size depending on if they are at the corner or adjacent to the center.",
#                 deduction=f"Based on the pattern that the circles alternate in size depending on if they are at the corner or adjacent to the center, the size of the missing part that is {answer_location} should be {answer}.",
#             ),
#             image,
#         )


class NumbersTrianglePattern(BaseModel):
    path_font: str = "fonts/OpenSans-Medium.ttf"
    image_size: int = 512
    scale_factor: int = 4
    num_sides: int = 3
    color: str = "#cfe2f3"  # Light blue

    def get_points(self, center: Point, radius: float) -> List[Point]:
        vertices = []
        for i in range(self.num_sides):
            theta = 2 * math.pi / self.num_sides * i
            x = center[0] + radius * math.cos(theta)
            y = center[1] + radius * math.sin(theta)
            vertices.append((x, y))
        return vertices

    def draw_circle(self, draw: ImageDraw, point: Point, radius: float, **kwargs):
        x, y = point
        position = x - radius, y - radius, x + radius, y + radius
        line_width = self.image_size * self.scale_factor // 200
        draw.ellipse(position, width=line_width, fill=self.color, **kwargs)

    def draw_text(self, draw: ImageDraw, point: Point, text: str):
        size = self.image_size * self.scale_factor
        draw.text(
            point,
            text=text,
            font=ImageFont.truetype(self.path_font, size=size // 14),
            anchor="mm",
            fill="black",
        )

    def make_sample(self, caption, current_option, answer, deduction, explanation):
        size = self.image_size * self.scale_factor
        image = Image.new("RGB", size=(size, size), color="white")
        draw = ImageDraw.Draw(image)
        center = size / 2, size / 2

        groups = extract_list_from_caption(caption)
        groups = [n if n != '?' else current_option for n in groups]

        groups = groups[::-1]
        numbers = groups[:3] + groups[3:6] + groups[6:]

        for i, point in enumerate(self.get_points(center, radius=size / 4)):
            # noinspection PyTypeChecker
            subpoints = self.get_points(point, radius=size / 10)
            draw.polygon(subpoints, outline="black", width=size // 200)
            for j, sub in enumerate(subpoints):
                # noinspection PyTypeChecker
                self.draw_circle(draw, sub, radius=size / 16, outline="black")
                # noinspection PyTypeChecker
                self.draw_text(draw, sub, str(numbers[i * 3 + j]))

        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        return image


def get_pixels(image: Image, fraction_x: float, fraction_y: float) -> Tuple[int, int]:
    x = round(image.width * fraction_x)
    y = round(image.height * fraction_y)
    return x, y


class VennPattern(BaseModel):
    path_template: str = "templates/puzzle-venn.png"
    path_font: str = "fonts/OpenSans-Medium.ttf"
    rule: str = "{} + {}"
    image_size: int = 512

    def draw_text(self, image: Image, text: str, position: Tuple[int, int]):
        draw = ImageDraw.Draw(image)
        draw.text(
            position,
            text,
            font=ImageFont.truetype(self.path_font, size=image.width // 16),
            anchor="mm",
            fill="black",
        )

    def make_sample(self, caption, current_option, answer, deduction, explanation):
        image = Image.open(self.path_template)

        options = extract_list_from_caption(caption)
        answer_i = options.index('?')
        [a, b, c] = [n if n != '?' else answer for n in options]

        ab = eval(self.rule.format(a, b))
        bc = eval(self.rule.format(b, c))

        if answer_i == 0:
            a = current_option
        else:
            c = current_option

        self.draw_text(image, str(a), get_pixels(image, 0.25, 0.5))
        self.draw_text(image, str(b), get_pixels(image, 0.50, 0.5))
        self.draw_text(image, str(c), get_pixels(image, 0.75, 0.5))
        self.draw_text(image, str(ab), get_pixels(image, 0.38, 0.5))
        self.draw_text(image, str(bc), get_pixels(image, 0.62, 0.5))

        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)

        return image


def select_pattern(name: str, **kwargs):

    if name == "circle_size_number":
        return CircleSizeNumberPattern(**kwargs)
    if name == "color_grid":
        return ColorGridPattern(**kwargs)
    if name == "color_hexagon":
        return ColorHexagonPattern(**kwargs)
    if name == "color_number_hexagon":
        return ColorNumberHexagonPattern(**kwargs)
    if name == "color_overlap_squares":
        return ColorOverlapSquaresPattern(**kwargs)
    if name == "color_size_circle":
        return ColorSizeCirclePattern(**kwargs)
    if name == "grid_number_color":
        return GridNumberColorPattern(**kwargs)
    if name == "grid_number":
        return GridNumberPattern(**kwargs)
    if name == "polygon_sides_color":
        return PolygonSidesColorPattern(**kwargs)
    if name == "polygon_sides_number":
        return PolygonSidesNumberPattern(**kwargs)
    if name == "rectangle_height_color":
        return RectangleHeightColorPattern(**kwargs)
    if name == "rectangle_height_number":
        return RectangleHeightNumberPattern(**kwargs)
    if name == "shape_morph":
        return ShapeMorphPattern(**kwargs)
    if name == "shape_reflect":
        return ShapeReflectPattern(**kwargs)
    if name == "shape_size_grid":
        return ShapeSizeGridPattern(**kwargs)
    if name == "shape_size_hexagon":
        return ShapeSizeHexagonPattern(**kwargs)
    if name == "size_cycle":
        return SizeCyclePattern(**kwargs)
    if name == "size_grid":
        return SizeGridPattern(**kwargs)
    if name == "triangle":
        return NumbersTrianglePattern(**kwargs)
    if name == "venn":
        return VennPattern(**kwargs)

    raise KeyError(name)


def sample_options(answer: str, options: List[str], k: int):
    # Ensure random order and no duplicates
    options = [o for o in options if o != answer]
    assert len(options) + 1 >= k
    options = random.sample(options, k=k - 1)
    options.append(answer)
    assert len(set(options)) == k
    return random.sample(options, k=k)


def generate_number_options(num: int, k: int) -> List[int]:
    # Automatically detect the range and random.sample
    assert num >= 0, "Negative numbers not supported yet"
    options = [10, 100, 1000, 10000, 100000]
    for max_value in options:
        if num <= max_value:
            values = [i for i in range(max_value) if i != num]
            lst = random.sample(values, k=k - 1)
            lst.append(num)
            assert len(set(lst)) == len(lst)
            return random.sample(lst, k=len(lst))
    raise ValueError(f"Range exceeded: {num}, options: {options}")


def convert_image_to_text(image: Image) -> str:
    # This is also how OpenAI encodes images: https://platform.openai.com/docs/guides/vision
    with io.BytesIO() as output:
        image.save(output, format="PNG")
        data = output.getvalue()
    return base64.b64encode(data).decode("utf-8")


def create_data(
    pattern_name: str,
    path: str = "data/train"
):
    # read data
    with open(f"{path}/{pattern_name}.json", "r") as f:
        samples = [json.loads(line) for line in f]

    os.makedirs(f"{path}/images_options/{pattern_name}", exist_ok=True)

    pattern = select_pattern(pattern_name)

    image_options = []
    for sample in tqdm(samples):
        try:
            answer = int(sample['answer'])
            options = [int(option) for option in sample['options']]
        except:
            answer = sample['answer']
            options = deepcopy(sample['options'])

        image_options_path = []
        for option_idx, option in enumerate(options):
            image = pattern.make_sample(
                caption=sample['caption'],
                current_option=option,
                answer=answer,
                deduction=sample['deduction'],
                explanation=sample['explanation']
            )

            image_path = sample['image'].replace('images/', 'images_options/').replace('.png', f'_{option_idx:02}.png')
            image.save(f"{path}/{image_path}")

            image_options_path.append(image_path)

        image_options.append({
            'image_options_path': image_options_path,
            'answer_index': options.index(answer),
        })

    with open(f"{path}/{pattern_name}_options.json", "w") as f:
        for line in image_options:
            f.write(json.dumps(line) + "\n")

if __name__ == "__main__":
    Fire()

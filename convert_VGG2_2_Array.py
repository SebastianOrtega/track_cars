import json


def get_polygon_mask(json_file, image_filename):
    with open(json_file, 'r') as file:
        data = json.load(file)

    if image_filename in data:
        regions = data[image_filename]["regions"]
        for region in regions.values():
            shape_attributes = region["shape_attributes"]
            if shape_attributes["name"] == "polygon":
                all_points_x = shape_attributes["all_points_x"]
                all_points_y = shape_attributes["all_points_y"]
                return [[int(x), int(y)] for x, y in zip(all_points_x, all_points_y)]

    return []


# Example usage
json_file = './labels_my-project-name_2024-07-14-09-49-52.json'
image_filename = 'first_frame.png'
mask_array = get_polygon_mask(json_file, image_filename)
print(mask_array)

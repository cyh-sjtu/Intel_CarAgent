"""
This functions modifies the original code from the following repository:
https://github.com/OpenRobotLab/VLM-Grounder?tab=readme-ov-file

Reference:
@inproceedings{xu2024vlmgrounder,
  title={VLM-Grounder: A VLM Agent for Zero-Shot 3D Visual Grounding},
  author={Xu, Runsen and Huang, Zhiwei and Wang, Tai and Chen, Yilun and Pang, Jiangmiao and Lin, Dahua},
  booktitle={CoRL},
  year={2024}
}

The original code is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License(https://creativecommons.org/licenses/by-nc-sa/4.0/).
"""

import glob
import math
import os
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def dynamic_stitch_images_fix_v2(
    images,
    fix_num=6,
    pre_resize=False,
    pre_resize_reference=512,
    annotate_id=True,
    ID_array=None,
    relative_ID_size=0.05,
    ID_color="black",
):
    """
    Stitch multiple images together in a grid layout based on the number of images.

    Args:
        images (list): A list of images to be stitched together.
        fix_num (int, optional): The number of images to be fixed in each grid layout. Defaults to 6.
        pre_resize (bool, optional): Whether to resize the images before stitching. Defaults to False.
        pre_resize_reference (int, optional): The reference size for pre-resizing the images. Defaults to 512.
        annotate_id (bool, optional): Whether to annotate the IDs on the stitched images. Defaults to True.
        ID_array (list, optional): A list of IDs corresponding to the images. Defaults to None.
        relative_ID_size (float, optional): The relative size of the ID annotation. Defaults to 0.05.
        ID_color (str, optional): The color of the ID annotation. Defaults to "black".

    Returns:
        list: A list of stitched images in a grid layout.
    """
    image_num = len(images)
    grid_images = []
    if image_num <= 4 * fix_num:  # use (4, 1)
        grid_images += stitch_images(
            images,
            (4, 1),
            pre_resize=pre_resize,
            pre_resize_reference=pre_resize_reference,
            annotate_id=annotate_id,
            ID_array=ID_array,
            relative_ID_size=relative_ID_size,
            ID_color=ID_color,
        )
    elif image_num <= 8 * fix_num:  # use (4, 1) and (2, 4)
        n_8 = math.ceil((image_num - 4 * fix_num) / 4)
        n_4 = fix_num - n_8
        if n_4 > 0:
            grid_images += stitch_images(
                images[: n_4 * 4],
                (4, 1),
                pre_resize=pre_resize,
                pre_resize_reference=pre_resize_reference,
                annotate_id=annotate_id,
                ID_array=ID_array[: n_4 * 4] if ID_array is not None else None,
                relative_ID_size=relative_ID_size,
                ID_color=ID_color,
            )
        if n_8 > 0:
            grid_images += stitch_images(
                images[n_4 * 4 :],
                (2, 4),
                pre_resize=pre_resize,
                pre_resize_reference=pre_resize_reference,
                annotate_id=annotate_id,
                ID_array=ID_array[n_4 * 4 :] if ID_array is not None else None,
                relative_ID_size=relative_ID_size,
                ID_color=ID_color,
            )
    elif image_num <= 16 * fix_num: # use (4, 1), (2, 4), (8, 2)
        n_16 = math.ceil((image_num - 8 * fix_num) / 8)  
        n_4_8 = fix_num - n_16
        tmp_num = max(image_num - n_16 * 16, 0)
        n_8 = math.ceil((tmp_num - 4 * n_4_8) / 4)
        n_4 = n_4_8 - n_8
        if n_4 > 0:
            grid_images += stitch_images(
                images[: n_4 * 4],
                (4, 1),
                pre_resize=pre_resize,
                pre_resize_reference=pre_resize_reference,
                annotate_id=annotate_id,
                ID_array=ID_array[: n_4 * 4] if ID_array is not None else None,
                relative_ID_size=relative_ID_size,
                ID_color=ID_color,
            )
        if n_8 > 0:
            grid_images += stitch_images(
                images[n_4 * 4 : n_4 * 4 + n_8 * 8],
                (2, 4),
                pre_resize=pre_resize,
                pre_resize_reference=pre_resize_reference,
                annotate_id=annotate_id,
                ID_array=ID_array[n_4 * 4 : n_4 * 4 + n_8 * 8]
                if ID_array is not None
                else None,
                relative_ID_size=relative_ID_size,
                ID_color=ID_color,
            )
        if n_16 > 0:
            grid_images += stitch_images(
                images[n_4 * 4 + n_8 * 8 :],
                (8, 2),
                pre_resize=pre_resize,
                pre_resize_reference=pre_resize_reference,
                annotate_id=annotate_id,
                ID_array=ID_array[n_4 * 4 + n_8 * 8 :]
                if ID_array is not None
                else None,
                relative_ID_size=relative_ID_size,
                ID_color=ID_color,
            )
    elif image_num <= 27 * fix_num: # use (4, 1), (2, 4), (8, 2), (9, 3)
        n_27 = math.ceil((image_num - 16 * fix_num) / 11)
        n_4_8_16 = fix_num - n_27
        tmp_num = max(image_num - n_27 * 27, 0)
        n_16 = math.ceil((tmp_num - 8 * n_4_8_16) / 8)
        n_4_8 = n_4_8_16 - n_16
        tmp_num = max(tmp_num - n_16 * 16, 0)
        n_8 = math.ceil((tmp_num - 4 * n_4_8) / 4)
        n_4 = n_4_8 - n_8
        if n_4 > 0:
            grid_images += stitch_images(
                images[: n_4 * 4],
                (4, 1),
                pre_resize=pre_resize,
                pre_resize_reference=pre_resize_reference,
                annotate_id=annotate_id,
                ID_array=ID_array[: n_4 * 4] if ID_array is not None else None,
                relative_ID_size=relative_ID_size,
                ID_color=ID_color,
            )
        if n_8 > 0:
            grid_images += stitch_images(
                images[n_4 * 4 : n_4 * 4 + n_8 * 8],
                (2, 4),
                pre_resize=pre_resize,
                pre_resize_reference=pre_resize_reference,
                annotate_id=annotate_id,
                ID_array=ID_array[n_4 * 4 : n_4 * 4 + n_8 * 8]
                if ID_array is not None
                else None,
                relative_ID_size=relative_ID_size,
                ID_color=ID_color,
            )
        if n_16 > 0:
            grid_images += stitch_images(
                images[n_4 * 4 + n_8 * 8 : n_4 * 4 + n_8 * 8 + n_16 * 16],
                (8, 2),
                pre_resize=pre_resize,
                pre_resize_reference=pre_resize_reference,
                annotate_id=annotate_id,
                ID_array=ID_array[n_4 * 4 + n_8 * 8 : n_4 * 4 + n_8 * 8 + n_16 * 16]
                if ID_array is not None
                else None,
                relative_ID_size=relative_ID_size,
                ID_color=ID_color,
            )
        if n_27 > 0:
            grid_images += stitch_images(
                images[n_4 * 4 + n_8 * 8 + n_16 * 16 :],
                (9, 3),
                pre_resize=pre_resize,
                pre_resize_reference=pre_resize_reference,
                annotate_id=annotate_id,
                ID_array=ID_array[n_4 * 4 + n_8 * 8 + n_16 * 16 :]
                if ID_array is not None
                else None,
                relative_ID_size=relative_ID_size,
                ID_color=ID_color,
            )
    else: # use more than fix_num images
        grid_images += stitch_images(
            images[: 27 * fix_num],
            (9, 3),
            pre_resize=pre_resize,
            pre_resize_reference=pre_resize_reference,
            annotate_id=annotate_id,
            ID_array=ID_array[: 27 * fix_num] if ID_array is not None else None,
            relative_ID_size=relative_ID_size,
            ID_color=ID_color,
        )
        grid_images += dynamic_stitch_images_fix_v2(
            images[27 * fix_num :],
            1,
            pre_resize,
            pre_resize_reference,
            annotate_id,
            ID_array[27 * fix_num :] if ID_array is not None else None,
            relative_ID_size,
            ID_color,
        )
    return grid_images


def stitch_images(
    images,
    grid_dims,
    pre_resize=False,
    pre_resize_reference=2048.0,
    annotate_id=True,
    ID_array=None,
    relative_ID_size=0.05,
    ID_color="black",
):
    """
    Stitch multiple images together into a grid.

    Args:
        images (List[str]): List of image paths or PIL Image objects.
        grid_dims (tuple[int, int]): Dimensions of the grid (rows, columns).
        pre_resize (bool, optional): Whether to resize images before stitching. Defaults to False.
        pre_resize_reference (float, optional): Reference size for resizing images. Defaults to 2048.0.
        annotate_id (bool, optional): Whether to annotate image IDs. Defaults to True.
        ID_array (List[int], optional): List of image IDs. Defaults to None.
        relative_ID_size (float, optional): Relative font size for image IDs. Defaults to 0.05.
        ID_color (str, optional): Color of the image ID annotation. Defaults to "black".

    Returns:
        List[Image.Image]: List of stitched grid images.
    """

    # if images[0] is str, then images should be loaded using PIL first
    if isinstance(images[0], str):
        images = [Image.open(img_path) for img_path in images]

    rows, cols = grid_dims
    if ID_array is None:
        ID_array = list(range(len(images)))
    else:
        # * assert length to be equal
        assert len(images) == len(
            ID_array
        ), "Images and ID_array should have the same length."

    # Calculate total number of images based on grid dimensions
    images_per_figure = rows * cols

    # Determine the total grid size
    total_width = images[0].size[0] * cols  # Total width of a row
    total_height = images[0].size[1] * rows  # Total height of a column
    longer_side = max(total_width, total_height)

    # Resize images if auto_resize is True
    if pre_resize and longer_side > pre_resize_reference:
        # Calculate the downsample ratio to make the longer side close to 2048 pixels
        resize_ratio = longer_side / pre_resize_reference
        images = [
            image.resize(
                (int(image.size[0] / resize_ratio), int(image.size[1] / resize_ratio))
            )
            for image in images
        ]
        # Prepare the list to hold all montage images
    grid_image_lists = []

    # Create montages
    for k in range(0, len(images), images_per_figure):
        # Extract the subset of images for current montage
        subset_images = images[k : k + images_per_figure]
        subset_ids = ID_array[k : k + images_per_figure]

        # Determine annotation font size relative to figure size
        relative_ID_size = 0.1  # Font size as a percentage of figure height

        # Determine figure size to be proportional to the number of images
        fig_width = cols * subset_images[0].size[0] / 100
        fig_height = (
            rows * subset_images[0].size[1] / 100
        )  # 100 is the default dpi of matplotlib, 1 inch = 100 pixels
        ID_size = subset_images[0].size[1] * relative_ID_size

        # Create a new figure for the montage with a smaller figure size
        fig, axs = plt.subplots(
            rows, cols, figsize=(fig_width, fig_height)
        )  # figsize is used with inch

        # Flatten the Axes array for easy iteration and indexing
        axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

        for idx, ax in enumerate(axs):
            # Remove axis for empty plots if the subset is smaller than grid size
            if idx >= len(subset_images):
                ax.axis("off")
                continue

            # Display image and remove axis
            ax.imshow(subset_images[idx])
            ax.axis("off")

            # Add annotation if required
            if annotate_id:
                ax.annotate(
                    str(subset_ids[idx]),
                    (5, 5),
                    color=ID_color,
                    fontsize=ID_size,
                    ha="left",
                    va="top",
                )

        # Adjust the layout to be tight
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

        # Save the montage figure to a bytes buffer
        buf = BytesIO()
        plt.savefig(buf, format="jpg", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)

        # Open the image for display and append to the list
        grid_image = Image.open(buf)
        grid_image_lists.append(grid_image)

    return grid_image_lists


if __name__ == "__main__":
    image_num = 80
    test_images_paths = glob.glob("/home/ao/Project/ImpressionMap/test_space/test_images_stitch_2025_04_15/*.png")
    test_images = [Image.open(path) for path in test_images_paths[:image_num]]
    print(f"Number of images: {len(test_images)}")
    ids = [0, 4]

    grid_images = dynamic_stitch_images_fix_v2(
        test_images, ID_array=ids, ID_color="red"
    )

    test_dir = "test_stitch"
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    for i, img in enumerate(grid_images):
        img.save(f"test_stitch/grid_{i}.jpg")

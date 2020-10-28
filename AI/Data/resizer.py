import os
from PIL import Image
from logging import error

imagesResized = list()

def resize(inDir, outDir, imgSize):
    image_dir = inDir
    output_dir = outDir
    resize_images(image_dir, output_dir, imgSize)
    return imagesResized

def resize_image(image, size):
    """Resize an image to the given size."""
    try:
        imagesResized.append(image)
        return image.resize(size, Image.ANTIALIAS)
    except ValueError as e:
        print("Could not resize image " + image)
        error(e)

def resize_images(image_dir, output_dir, size):
    """Resize the images in 'image_dir' and save into 'output_dir'."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(image_dir)
    num_images = len(images)
    for i, image in enumerate(images):
        pathPreview = str(output_dir) + "/" + image
        if os.path.exists(pathPreview):
            print("Skipping " + pathPreview)
            continue
        with open(os.path.join(image_dir, image), 'r+b') as f:
            with Image.open(f) as img:
                width, height = img.size
                if width is not size and height is not size:
                    try:
                        img = resize_image(img, [size, size])
                        img.save(os.path.join(output_dir, image), img.format)
                    except OSError as e:
                        print("Error. Could not save file")
        if (i+1) % 100 == 0:
            print ("[{}/{}] Resized {} image and saved into '{}'. {}%"
                   .format(i+1, num_images, image, output_dir, round(i/num_images)))
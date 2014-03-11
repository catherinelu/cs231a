from PIL import Image, ImageDraw
import random
import pdb

WHITE_PIXEL = 255
WHITE_THRESHOLD = 200
NUM_CHAR_THRESHOLD = 40
MAX_WIDTH = 400
MAX_HEIGHT = 100

def create_cluster(cur_x, cur_y, group, non_white_pixels, width, height):
    if cur_x < 0 or cur_x >= width or cur_y < 0 or cur_y >= height:
        return
    if (cur_x, cur_y) in non_white_pixels:
        group.append((cur_x, cur_y))
        non_white_pixels.remove((cur_x, cur_y))
        for x in xrange(-1, 2):
            for y in xrange(-1, 2):
                if bool(x == 0) != bool(y == 0):  # XOR
                    create_cluster(cur_x + x, cur_y + y, group, non_white_pixels, width, height)


def main():
    img = Image.open('images/andy.png')

    width, height = img.size[0], img.size[1]

    if width > MAX_WIDTH:
        width = MAX_WIDTH
        height = int(MAX_WIDTH / float(width) * height)
    if height > MAX_HEIGHT:
        height = MAX_HEIGHT
        width = int(MAX_HEIGHT / float(height) * width)

    img = img.resize((width, height),Image.ANTIALIAS)
    pixels = img.load()

    # First, put the indices of all the non-white pixels into a set
    non_white_pixels = set()
    for x in range(width):
        for y in range(height):
            if pixels[x, y] < WHITE_THRESHOLD:
                non_white_pixels.add((x, y))

    # Next, cluster pixels into groups
    characters = []  # List of lists of character groupings
    while len(non_white_pixels) > 0:
        non_white_pixel = non_white_pixels.pop()
        group = [non_white_pixel]

        old_x = non_white_pixel[0]
        old_y = non_white_pixel[1]

        for x in range(-1, 2):  # Goes from -1 to 1
            for y in range(-1, 2):  # Goes from -1 to 1
                if bool(x == 0) != bool(y == 0):  # XOR
                    create_cluster(old_x + x, old_y + y, group, non_white_pixels, width, height)
        characters.append(group)

    # White out all of the pixels
    for x in range(width):
        for y in range(height):
            pixels[x, y] = WHITE_PIXEL

    # Get rid of characters that do not have enough pixels (likely to not be an actual character)
    significant_characters = []
    for character in characters:
        if len(character) > NUM_CHAR_THRESHOLD:
            significant_characters.append(character)
    characters = significant_characters

    # Sort the characters based on the smallest x, y coordinate (which is smallest x)
    characters.sort(key=min)

    for i, character in enumerate(characters):
        color = 100 if i % 2 == 0 else 0
        for pixel in character:
            pixels[pixel[0], pixel[1]] = color

    print 'Num segmented characters:', len(characters)
    img.show()
    img.save('adam-segmentation.png')

if __name__ == '__main__':
    main()

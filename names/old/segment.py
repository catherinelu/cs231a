from PIL import Image, ImageDraw

WHITE_PIXEL = 255
WHITE_THRESHOLD = 50

def main():
    img = Image.open('images/alp.png')
    pixels = img.load()
    width, height = img.size[0], img.size[1]

    # x_set = set()
    # for x in range(width):
    #     for y in range(height):
    #         if pixels[x, y] < WHITE_THRESHOLD:
    #             x_set.add(x)

    # print x_set

    new_character = True
    min_x = None
    max_x = None
    characters = []
    for x in range(width):
        is_nonwhite_pixel_in_column = False
        count = 0

        for y in range(height):
            if pixels[x, y] < WHITE_THRESHOLD:
                if count < 1:
                    count += 1
                    continue
                print 'Non-white pixel found'
                if new_character:
                    min_x = x  # If min_x is not set, it's the first one found
                else:
                    max_x = x  # Set max_x
                is_nonwhite_pixel_in_column = True

        if is_nonwhite_pixel_in_column:  # The column contains black
            if new_character:  # It is the beginning of a new character
                new_character = False
            else:  # It is the continuation of a new character. Do nothing.
                pass
        else:  # The column contains all white pixels
            if new_character:
                pass
            else:  # This marks the end of a character
                characters.append((min_x, max_x))
                new_character = True
        

    print characters

    for character in characters:
        draw = ImageDraw.Draw(img)
        draw.line((character[0], 0, character[0], height), fill=128, width=2)
        draw.line((character[1], 0, character[1], height), fill=128, width=2)
    img.show()
    img.save('adam-segmentation.png')

if __name__ == '__main__':
    main()

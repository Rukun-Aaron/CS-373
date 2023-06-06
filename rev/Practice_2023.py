def flipImageAlongVerticalCenterColumn(pixel_array, image_width, image_height):
    new_image = createInitializedGreyscalePixelArray(image_width, image_height, 0)
    for row in range(image_height):
        for col in range(image_width):
            new_image[row][col] = pixel_array[row][image_width - col - 1]
    return new_image

def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):

    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array

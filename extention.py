# Built in packages
import math
import sys
from pathlib import Path

# Matplotlib will need to be installed if it isn't already. This is the only package allowed for this base part of the 
# assignment.
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import matplotlib.image as mpimg


# import our basic, light-weight png reader library
import imageIO.png

# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):

    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    image = mpimg.imread(input_filename) 
  
    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b,image)


# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):

    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array


# You can add your own functions here:
def computeRGBToGreyscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):
    
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    
    for i in range(image_height):
        for j in range(image_width):
            
                greyscale_value = round(0.299 * pixel_array_r[i][j] + 0.587 * pixel_array_g[i][j] + 0.114 * pixel_array_b[i][j])
                greyscale_pixel_array[i][j] = greyscale_value
           
    return greyscale_pixel_array

def scaleTo0And255AndQuantize(pixel_array, image_width, image_height):

    min_value, max_value = computeMinAndMaxValues(pixel_array, image_width, image_height)
    scaled_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)

    for i in range(image_height):
        for j in range(image_width):
            if max_value == min_value:
                scaled_pixel_array[i][j] = 0
            else:

                strechted_value = round((pixel_array[i][j] - min_value) * (255-0) / (max_value - min_value)) +0
                if strechted_value > 255:
                    strechted_value = 255
                elif strechted_value < 0:
                    strechted_value = 0
                else :
                    scaled_pixel_array[i][j] = strechted_value
    return scaled_pixel_array
def computeMinAndMaxValues(pixel_array, image_width, image_height):

    min_value = min([min(row) for row in pixel_array])
    max_value = max([max(row) for row in pixel_array])
    return min_value, max_value

def computeHorizontalEdgesSobelAbsolute(pixel_array, image_width, image_height):

    new_image = createInitializedGreyscalePixelArray(image_width, image_height, 0.0)


    for j in range(1, image_height - 1):
        for i in range(1, image_width - 1):

            value = (
                pixel_array[j - 1][i - 1] + (pixel_array[j - 1][i])* 2 + pixel_array[j - 1][1 + i] -
                pixel_array[1 + j][i - 1] -(pixel_array[1 + j][i]) * 2 - pixel_array[1 + j][1 + i])
            value = abs(value / 8.0)


            new_image[j][i] = value

    return new_image
def computeVerticalEdgesSobelAbsolute(pixel_array, image_width, image_height):
    
    new_image = createInitializedGreyscalePixelArray(image_width, image_height,0.0)
    # test_image = createInitializedGreyscalePixelArray(image_width, image_height,0.0)
    for i in range(1, image_height - 1):
        for k in range(1, image_width - 1):
            answer = pixel_array[i-1][1+k] + (pixel_array[i][1 + k]) * 2 + pixel_array[1 + i][1 + k] - pixel_array[i - 1][k - 1] - (pixel_array[i][k - 1]) * 2 - pixel_array[1 + i][k - 1]

            new_image[i][k] = abs(float(answer/8.0))
    return new_image
def computeEdgeMagnitude(horizontal, vertical, image_width, image_height):
    new_array = createInitializedGreyscalePixelArray(image_width, image_height)
    
    # the sizes of the 'horizontal' and 'vertical' pixel arrays should be the same
       # the height of horizontal/vertical == image_height (ROWS)
       # the width of horizontal/vertical == image_width (COLUMNS)
    
    for i in range(image_height):
        for j in range(image_width):
            mag = math.sqrt(math.pow(horizontal[i][j],2) + math.pow(vertical[i][j],2))
            
            new_array[i][j] = mag

    return new_array
def computeGaussianAveraging3x3RepeatBorder(pixel_array, image_width, image_height):

    new_list = createInitializedGreyscalePixelArray(image_width, image_height)

    kernel = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]


    for i in range(image_height):
        for j in range(image_width):

            count = 0

            # Iterate over the kernel
            for k in range(-1, 2):
                for l in range(-1, 2):
                    val1 = min(max(i + k, 0), image_height - 1)
                    val2 = min(max(j + l, 0), image_width - 1)
                    count += pixel_array[val1][val2] * kernel[1+ k][1+l]

            new_list[i][j] = count / 16

    return new_list
def computeStandardDeviationImage3x3(pixel_array, image_width, image_height):

    new_image = createInitializedGreyscalePixelArray(image_width, image_height, 0.0)


    for i in range(1, image_height - 1):
        for j in range(1, image_width - 1):

            new_array = []
            for k in range(-1, 2):
                for l in range(-1, 2):
                    first = i + k
                    second = j + l
                    new_array.append(pixel_array[first][second])

            temp_value1 = 0

            for x in new_array:
                temp_value1 += (x - (sum(new_array) / 9)) ** 2
            val2 = temp_value1 / 9
            final_val = math.sqrt(val2)

            new_image[i][j] = final_val

    return new_image
def computeStandardDeviationImage5x5(pixel_array, image_width, image_height):
    
        new_image = createInitializedGreyscalePixelArray(image_width, image_height, 0.0)
    
    
        for i in range(2, image_height - 2):
            for j in range(2, image_width - 2):
    
                new_array = []
                for k in range(-2, 3):
                    for l in range(-2, 3):
                        first = i + k
                        second = j + l
                        new_array.append(pixel_array[first][second])
    
                temp_value1 = 0
    
                for x in new_array:
                    temp_value1 += (x - (sum(new_array) / 25)) ** 2
                val2 = temp_value1 / 25
                final_val = math.sqrt(val2)
    
                new_image[i][j] = final_val
    
        return new_image
def computeThresholdGE(pixel_array, threshold_value, image_width, image_height):
    new_array = createInitializedGreyscalePixelArray(image_width, image_height)
    
    for i in range(image_height):
        for j in range(image_width):
            if pixel_array[i][j] >= threshold_value:
                
                new_array[i][j] = 255

    return new_array
def computeErosion8Nbh5x5FlatSE(pixel_array, image_width, image_height):
    
    erosion_image = createInitializedGreyscalePixelArray(image_width, image_height)

    for row in range(image_height):
        for column in range(image_width):
            if (
                row > 1 and row < image_height - 2 and
                column > 1 and column < image_width - 2 and
                pixel_array[row-2][column] > 0 and
                pixel_array[row+2][column] > 0 and
                pixel_array[row][column-2] > 0 and
                pixel_array[row][column+2] > 0 and
                pixel_array[row-2][column-2] > 0 and
                pixel_array[row-2][column+2] > 0 and
                pixel_array[row+2][column-2] > 0 and
                pixel_array[row+2][column+2] > 0 and
                pixel_array[row-1][column-1] > 0 and
                pixel_array[row-1][column+1] > 0 and
                pixel_array[row+1][column-1] > 0 and
                pixel_array[row+1][column+1] > 0 and
                pixel_array[row-1][column-2] > 0 and
                pixel_array[row-1][column+2] > 0 and
                pixel_array[row+1][column-2] > 0 and
                pixel_array[row+1][column+2] > 0 and
                pixel_array[row-2][column-1] > 0 and
                pixel_array[row-2][column+1] > 0 and
                pixel_array[row+2][column-1] > 0 and
                pixel_array[row+2][column+1] > 0
            ):
                erosion_image[row][column] = 1

    return erosion_image
def computeDilation8Nbh5x5FlatSE(pixel_array, image_width, image_height):
    dilated_image = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(2, image_height - 2):
        for j in range(2, image_width - 2):
            for row in range(-2, 3):
                for col in range(-2, 3):
                    if (row + i >= 0 and row + i < image_height and col + j >= 0 and col + j < image_width and pixel_array[row + i][col + j] != 0):
                        dilated_image[i][j] = 1
    return dilated_image
class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)
def computeConnectedComponentLabeling(pixel_array, image_width, image_height):
    visited=createInitializedGreyscalePixelArray(image_width, image_height)
    ccimg= createInitializedGreyscalePixelArray(image_width, image_height)

    ccdict = dict()
    ccount = 1

    for i in range(image_height):
        for j in range(image_width):
            
            if visited[i][j] == 0 and pixel_array[i][j] !=0:
                number = bfs_traversal(pixel_array, visited, i, j, image_width, image_height, ccimg, ccount)
                ccdict[ccount] = number
                ccount += 1
    return ccimg, ccdict

def bfs_traversal(pixel_array, visited, i, j, width, height, ccimg, count):

    queue = Queue()
    visited[i][j] = 1
    queue.enqueue((i,j))
    pixel_number = 0
    neighbour_coordinates = [(-1, 0), (0,-1), (0,1), (1,0)]
    while(not queue.isEmpty()):
        x,y = queue.dequeue()
        ccimg[x][y] = count
        pixel_number += 1

        for neighbour in neighbour_coordinates:
            x1 = x + neighbour[0]
            y1 = y + neighbour[1]
            if (x1 >= 0 and x1 < height and y1 >= 0 and y1 < width and visited[x1][y1] == 0 and pixel_array[x1][y1] != 0):
                queue.enqueue((x1,y1))
                visited[x1][y1] = 1
    
    return pixel_number

def computeBoundaryBox(ccimg, ccdict, image_width, image_height):
    largest_component = max(ccdict, key=ccdict.get)
    # sort ccdict by keyvalue

    barcode_box = []
    density = 0.70
  
    sorted_ccsizes = sorted(ccdict.items(), key=lambda x: x[1], reverse=True)

    for key, value in sorted_ccsizes:
           

            min_x = image_width
            min_y = image_height
            max_x = 0
            max_y = 0

            for row in range(image_height):
                for col in range(image_width):
                    if ccimg[row][col] == key:
                        min_x = min(min_x, col)
                        min_y = min(min_y, row)
                        max_x = max(max_x, col)
                        max_y = max(max_y, row)
                
            side1 = max_x - min_x
            side2 = max_y - min_y
            length = max(side1, side2)
            breadth = min(side1, side2)
            temparea = length * breadth
            tempdensity = value / temparea
          
            ratio = length / breadth
            if  density<tempdensity and ((length >50)):
                barcode_box.append([min_x, min_y, max_x, max_y])
                # area = temparea
                    
    return barcode_box




# This is our code skeleton that performs the barcode detection.
# Feel free to try it on your own images of barcodes, but keep in mind that with our algorithm developed in this assignment,
# we won't detect arbitrary or difficult to detect barcodes!



def main():

    command_line_arguments = sys.argv[1:]

    SHOW_DEBUG_FIGURES = True

    # this is the default input image filename
    filename = "Multiple_barcodes3"
    input_filename = "images/"+filename+".png"

    if command_line_arguments != []:
        input_filename = command_line_arguments[0]
        SHOW_DEBUG_FIGURES = False

    output_path = Path("output_images")
    if not output_path.exists():
        # create output directory
        output_path.mkdir(parents=True, exist_ok=True)

    output_filename = output_path / Path(filename+"_output.png")
    if len(command_line_arguments) == 2:
        output_filename = Path(command_line_arguments[1])

    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b, orignal_image) = readRGBImageToSeparatePixelArrays(input_filename)
 
    # setup the plots for intermediate results in a figure
   
    fig1, axs1 = pyplot.subplots(2, 2,figsize=(8, 8))
    fig1.set_size_inches(10, 8)
    axs1[0, 0].set_title('Input red channel of image')
    axs1[0, 0].imshow(px_array_r, cmap='gray')
    axs1[0, 1].set_title('Input green channel of image')
    axs1[0, 1].imshow(px_array_g, cmap='gray')
    axs1[1, 0].set_title('Input blue channel of image')
    axs1[1, 0].imshow(px_array_b, cmap='gray')


    # STUDENT IMPLEMENTATION here

    # convert the image to greyscale and normalize 
    greyscale = computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)
    scaled_greyscale  = scaleTo0And255AndQuantize(greyscale, image_width, image_height)

    # horizontal_edges = computeHorizontalEdgesSobelAbsolute(scaled_greyscale, image_width, image_height)
    # verticle_edges = computeVerticalEdgesSobelAbsolute(scaled_greyscale, image_width, image_height)
    # edge_magnitude = computeEdgeMagnitude(horizontal_edges, verticle_edges, image_width, image_height)

    # blurred_image = computeGaussianAveraging3x3RepeatBorder(edge_magnitude, image_width, image_height)
    #Image gradient 
    #apply standard deviation 
    standard_deviation = computeStandardDeviationImage5x5(scaled_greyscale, image_width, image_height)
    #Apply Gaussian blur
    for i in range(4):
        box_filter = computeGaussianAveraging3x3RepeatBorder(standard_deviation, image_width, image_height) # returns new greyscale array
        standard_deviation = box_filter
   
    convert_to_binary = computeThresholdGE(standard_deviation, 25, image_width, image_height)

    for j in range(3):
        eroded_image = computeErosion8Nbh5x5FlatSE(convert_to_binary, image_width, image_height)
        convert_to_binary = eroded_image
    
    for k in range(6):
        dialated_image = computeDilation8Nbh5x5FlatSE(eroded_image, image_width, image_height)
        eroded_image = dialated_image
    ccimage ,ccsizes= computeConnectedComponentLabeling(dialated_image, image_width, image_height)
    boxes = computeBoundaryBox(ccimage, ccsizes,image_width, image_height)
    
    px_array = orignal_image
 

    # The following code is used to plot the bounding box and generate an output for marking
    # Draw a bounding box as a rectangle into the input image
    axs1[1, 1].set_title('Final image of detection')
    axs1[1, 1].imshow(px_array, cmap='gray')
    for box in boxes:
        bbox_min_x,bbox_min_y,bbox_max_x,bbox_max_y = box 
        rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=1,
                        edgecolor='g', facecolor='none')
        axs1[1, 1].add_patch(rect)

    # write the output image into output_filename, using the matplotlib savefig method
    extent = axs1[1, 1].get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)

    if SHOW_DEBUG_FIGURES:
        # plot the current figure
        pyplot.show()



main()
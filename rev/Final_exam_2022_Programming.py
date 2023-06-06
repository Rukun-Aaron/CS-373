def computeInterquartileRange3x3(pixel_array, image_width, image_height):
    list = [*range(1, image_height ,1)]
    print( list)
    new_image = createInitializedGreyscalePixelArray(image_width, image_height, 0)
    for row in range(image_height):
        if row !=0 and row != image_height - 1 :
            for col in range(image_width):
                if col != 0 and col!= image_width - 1 :
                    new_image[row][col] = pixel_array[row][col]
                    neighbors = [pixel_array[row-1][col-1] ,pixel_array[row-1][col] ,pixel_array[row-1][col+1] ,pixel_array[row][col-1] ,pixel_array[row][col],pixel_array[row][col+1] ,pixel_array[row+1][col-1] ,pixel_array[row+1][col] ,pixel_array[row+1][col+1]]
                    neighbors.sort()
                    inter_quartile = get_interquartile_range(neighbors)
                    new_image[row][col] = inter_quartile
    return new_image
def get_median(array):
    
        array.sort()
        half = len(array) // 2
        
        if not len(array) % 2:
           
            return (array[half - 1] + array[half]) / 2.0
        return array[half]
def get_interquartile_range(array):
    array.sort()
    half = len(array) // 2
    if not len(array) % 2:
        Q1 = get_median(array[:half])
        Q3 = get_median(array[half:])
    else:
        Q1 = get_median(array[:half])
        Q3 = get_median(array[half + 1:])
    return Q3 - Q1

def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):

    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array

# image_width = 6
# image_height = 5
# pixel_array = [ [6, 3, 2, 6, 4, 7], 
#                 [5, 3, 2, 7, 0, 6], 
#                 [6, 2, 7, 7, 1, 7], 
#                 [7, 6, 6, 2, 7, 3], 
#                 [2, 2, 2, 5, 1, 2] ]
# print(computeInterquartileRange3x3(pixel_array, image_width, image_height))

def computeCumulativeHistogram(pixel_array, image_width, image_height, nr_bins = 256):

    # compute histogram
    histogram = [0.0 for q in range(nr_bins)]
    
    for y in range(image_height):
        for x in range(image_width):
            histogram[pixel_array[y][x]] += 1.0


    # compute cumulative histogram
    cumulative_histogram = [0.0 for q in range(nr_bins)]

    running_sum = 0.0
    for q in range(nr_bins):
        running_sum += histogram[q]
        cumulative_histogram[q] = running_sum

    return cumulative_histogram

def computeLookupTableHistEq(pixel_array, image_width, image_height, nr_bins):
    equalized_histogram = []
    historgram = computeCumulativeHistogram(pixel_array, image_width, image_height, nr_bins)
    
    # min_count = min(historgram)
    min_count = next((i for i, x in enumerate(historgram) if x), None)
    max_count = max(historgram)
    
    max_value = nr_bins-1
    
    for num in historgram:
        

        equalized_value = max_value * ((num - min_count) /(max_count - min_count))
        equalized_histogram.append(equalized_value)
    
    return equalized_histogram
image_width = 6
image_height = 5
pixel_array = [ [6, 3, 2, 6, 4, 7], 
                [5, 3, 2, 7, 0, 6], 
                [6, 2, 7, 7, 1, 7], 
                [7, 6, 6, 2, 7, 3], 
                [2, 2, 2, 5, 1, 2] ]
nr_bins = 8
lookup_table= computeLookupTableHistEq(pixel_array, image_width, image_height, nr_bins)
for q in range(len(lookup_table)):
   print("{}: {}".format(q, round(lookup_table[q], 2)))
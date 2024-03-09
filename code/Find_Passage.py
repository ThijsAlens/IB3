def Find_Passage(data, size, threshold, min_area_size):
    """
    Find_Passage searches for any possible opening in the data (like an open door).

    Args:
        data : The frame that needs to be processed.
        size : The size of the areas that are seen as 1. (Ex.: a 10x10 and a size of 2 will check every other 2x2 area)
        threshold : The maximum value an area can be to still be seen as an opening (typically low)
        min_area_size : The minimal total size of an area. (Ex.: a 10x10 and a min_area_size of 9 will consider an area that is at least 3x3)
        
    Returns:
        list : area found : [ [starting row, starting column] , [ending row, ending column] ].
               area not found : []

    Recomended input:
        data = data
        size (not quantizesed) = 10 | size (quantizised) = 1
        threshold = 20
        min_area_size = 100
    """

    significant_areas = []
    for row in range(0, len(data)-size, size):
        first_area_on_row = True
        curr_length = len(significant_areas)
        for column in range(0, len(data[row])-size, size):
            area_mean = np.mean(data[row:(row+size), column:(column+size)])
            if (area_mean < threshold and first_area_on_row):
                significant_areas.append([row, [column]])
                first_area_on_row = False
            if (area_mean < threshold and not first_area_on_row):
                significant_areas[-1][1].append(column)
        if (curr_length != len(significant_areas)):
            significant_areas[-1][1] = [min(significant_areas[-1][1]), max(significant_areas[-1][1])]
    print(significant_areas)

    mean_minimum = 0
    mean_maximum = 0
    for i in range(len(significant_areas)):
        mean_minimum += significant_areas[i][1][0]
        mean_maximum += significant_areas[i][1][1]
    mean_maximum /= len(significant_areas)
    mean_minimum /= len(significant_areas)
    if (mean_maximum-mean_minimum > min_area_size**0.5 and significant_areas[-1][0]-significant_areas[0][0] > min_area_size**0.5):
        return [[significant_areas[0][0], mean_minimum], [significant_areas[-1][0], mean_maximum]]
    else:
        return []
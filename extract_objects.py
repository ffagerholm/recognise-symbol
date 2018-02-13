from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square


def get_object_boudaries(image, area_treshold=100):
    # apply threshold
    thresh = threshold_otsu(image)
    bw = closing(image > thresh, square(3))

    # remove artifacts connected to image border
    cleared = clear_border(bw)

    # label image regions
    label_image = label(cleared)

    # compile list of boundaries for found objects
    object_boudaries = []
    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= area_treshold:
            # region.bbox gives a 4-tuple of: 
            # starting index in row and column,
            # ending index in row and column
            minr, minc, maxr, maxc = region.bbox
            object_boudaries.append([slice(minr, maxr), slice(minc, maxc)])

    return object_boudaries
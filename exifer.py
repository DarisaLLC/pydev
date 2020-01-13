
from exif import Image
from PIL import Image
from PIL.ExifTags import TAGS
import sys
from pathlib import Path
import cameratransform

def get_exif(fn):
    ret = {}
    i = Image.open(fn)
    info = i._getexif()
    for tag, value in info.items():
        decoded = TAGS.get(tag, tag)
        ret[decoded] = value
    return ret

def main(fqfn):

    camp = cameratransform.getCameraParametersFromExif(fqfn, verbose=True)
    return camp

    # # read the exif information of the file
    # exif = get_exif(fqfn)
    # # get the focal length
    # f = exif["FocalLength"][0] / exif["FocalLength"][1]
    # # TBD: cache in dbase
    # # no dbase so get it from the exif information
    # if not sensor_size or sensor_size is None:
    #     sensor_size = (
    #         exif["ExifImageWidth"] / (exif["FocalPlaneXResolution"][0] / exif["FocalPlaneXResolution"][1]) * 25.4,
    #         exif["ExifImageHeight"] / (exif["FocalPlaneYResolution"][0] / exif["FocalPlaneYResolution"][1]) * 25.4)
    # # get the image size
    # image_size = (exif["ExifImageWidth"], exif["ExifImageHeight"])
    # # print the output if desired
    # if verbose:
    #     print("Intrinsic parameters for '%s':" % exif["Model"])
    #     print("   focal length: %.1f mm" % f)
    #     print("   sensor size: %.1f mm × %.1f mm" % sensor_size)
    #     print("   image size: %d × %d Pixels" % image_size)
    # return f, sensor_size, image_size


if __name__ == '__main__':
    argcnt = len(sys.argv)
    if argcnt < 2 or (not Path(sys.argv[1]).is_file() or not Path(sys.argv[1]).exists()):
        print(' File Does not exist or found ')
        sys.exit(1)

    main(sys.argv[1])

import sys
from pathlib import Path
import skvideo.utils
import skvideo.io
import skvideo.datasets
import json

def getMetaData(filename):
    if not Path(filename).exists() or not Path(filename).is_file():
        return None

    metadata = skvideo.io.ffprobe(filename)
    return metadata

if __name__ == '__main__':
    filename = skvideo.datasets.bigbuckbunny()
    if len(sys.argv) >= 2 and Path(sys.argv[1]).exists() and Path(sys.argv[1]).is_file():
        filename = sys.argv[1]

    metadata = getMetaData(filename)
    print(metadata.keys())
    print(json.dumps(metadata["video"], indent=4))

    # # here you can set keys and values for parameters in ffmpeg
    # inputparameters = {}
    # outputparameters = {}
    # reader = skvideo.io.FFmpegReader(filename,
    #                                  inputdict=inputparameters,
    #                                  outputdict=outputparameters)
    #
    # # iterate through the frames
    # accumulation = 0
    # for frame in reader.nextFrame():
    #         # do something with the ndarray frame
    #         accumulation += np.sum(frame)

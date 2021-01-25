# Copyright 2019 Cyril Poulet, cyril.poulet@centraliens.net
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import cv2
import logging
import sys

from detector import DetectorFactory

logging.basicConfig(stream=sys.stdout,
                    level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class VideoAnalyzer(object):

    def __init__(self, detectors=None, frame_ratio=1.):
        """
        This class instantiate N detectors and applies them to frames of a given video

        :param detectors: [(detector_name, parameters_dict), ...] -> list of tuples (str, dict). The detector_name must be an existing Detector subclass
        :param frame_ratio: ratio of frames to analyze (1 -> all frames, 0.2 -> 1 on 5, etc)
        """
        if detectors is None:
            detectors = [("HumanDetector", {})]

        self._logger = logging.getLogger("VideoAnalyzer")
        self._logger.info("Creating Video Analyzer")
        self._detectors_names = []
        self._detectors = []

        for key, vals in detectors:
            self._logger.info('Instantiating detector {}'.format(key))
            self._detectors_names.append(key)
            self._detectors.append(DetectorFactory.get_detector(key)(**vals))

        self._analysis_ratio = float(frame_ratio)

    def analyze_video(self, path_to_video):
        """
        Loads a video and applies the detectors to the frames, with respect to the ratio defined at instantiation

        :param path_to_video: path to the video to annalyze
        :return:  {
                    "fps": vid_fps, 
                    "codec_code": vid_codec_code, 
                    "frames": list of dict {
                                    "frame_index": ,
                                    "frame_timestamp": 
                                    "detector_name_1":  {detection results for given frame},
                                    ...
                                    }
                  }
        """
        one_frame_every_n_frame = int(1./self._analysis_ratio)
        self._logger.info("Analyzing file {}, 1 frame every {} frame".format(path_to_video, one_frame_every_n_frame))
        # open video
        cap = cv2.VideoCapture(path_to_video)
        # get various infos on the video
        vid_fps = cap.get(cv2.CAP_PROP_FPS)
        vid_codec_code = cap.get(cv2.CAP_PROP_FOURCC)
        self._logger.info("Detected codec and FPS: {}, {}".format(vid_codec_code, vid_fps))

        # determine processing batch size from detectors
        max_batch_size = min([c.batch_max_size for c in self._detectors])
        current_frame_ind = 0
        input_timestamps = []
        input_images = []
        frame_results = []

        def process_results(frames_info, results_dict):
            """
            merge results from various detectors

            :param frames_info: list of tuples (index of frame, time of frame)
            :param results_dict: list of dict {detector_name: {detection results for given frame}}
            :returns: list of dict {
                                    "frame_index": ,
                                    "frame_timestamp": 
                                    "detector_name_1":  {detection results for given frame},
                                    ...
                                    }
            """
            results = []
            for i, (f_ind, f_tsp) in enumerate(frames_info):
                im_res = {
                    "frame_index": f_ind,
                    "frame_timestamp": f_tsp
                }
                for key in results_dict:
                    im_res[key] = results_dict[key][i]
                results.append(im_res)
            return results

        while True:
            r, img = cap.read()
            if not r:
                # we reached the end of the video
                break

            # count frames, skip frames if necessary to respect processing ratio
            current_frame_ind += 1
            if (current_frame_ind - 1) % one_frame_every_n_frame != 0:
                continue

            if len(input_images) < max_batch_size:
                # while we do not have a complete batch, stack images to process
                input_images.append(img)
                input_timestamps.append((current_frame_ind, cap.get(cv2.CAP_PROP_POS_MSEC)))

            if len(input_images) == max_batch_size:
                # process batch
                detection_results = {}
                for det in self._detectors:
                    det_results = det.analyze_images(input_images)
                    detection_results[det.detected_category] = det_results
                batch_results = process_results(input_timestamps, detection_results)
                # store results
                frame_results.extend(batch_results)
                input_images = []
                input_timestamps = []

        # process last batch (which may be incomplete)
        detection_results = {}
        if input_images:
            for det in self._detectors:
                det_results = det.analyze_images(input_images)
                detection_results[det.detected_category] = det_results
            batch_results = process_results(input_timestamps, detection_results)
            frame_results.extend(batch_results)

        cap.release()
        self._logger.info("Analyzed {} images".format(len(frame_results)))

        return {"fps": vid_fps, "codec_code": vid_codec_code, "frames": frame_results}

    def close(self):
        self._logger.info("Closing all detectors")
        [c.close() for c in self._detectors]


if __name__ == "__main__":
    import json
    with open('variables.json') as f:
        params = json.load(f)

    test_video = "example/derby_testmatch_1_firstjam.mp4"

    analyzer = VideoAnalyzer(**params["human_detection"])
    results = analyzer.analyze_video(test_video)
    with open('test.json', 'w') as f:
        json.dump(results, f)

    analyzer.close()

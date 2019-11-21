from __future__ import print_function
# if __name__ == '__main__' and __package__ == None:
if __name__ == '__main__':
    # __package__ = 'imagemp'
    import ctypes           # need to define the types for the data in the memory shared between processes
    import argparse, os  # provide interface for calling this script
    import imagemp as imp
    from simple_display import SimpleDisplay as imp_SimpleDisplay
    from recorder import Recorder as imp_Recorder
    from tracker_pad import TrackPad as imp_Analysis

    from imagemp.shframe_grabbers.opencv_grabber import FrameGrabberCV2Camera

    output_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), 'output'))

    import time

    # Get input parameters
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=False, help="path to input video file")
    ap.add_argument("-r", "--recvid", required=False, help="path to recorded video file")
    args = vars(ap.parse_args())
    file_folder = os.path.dirname(os.path.realpath(__file__))
    vid_filename = args['video'] if args['video'] is not None else \
        os.path.abspath(os.path.join(file_folder,
                                     r'../projects/wiic/video/2019_10_31-3/Night/Front/Straight/2019-10-31 17-45-42.mp4'))
    vid_av_filename = os.path.abspath(os.path.join(output_folder, '_output.npy'))
    source = 'file'   # 'camera' or 'file'

    # Get image shape:
    # In order to determine the frame size, the VideoCapture should be started. However, the generated object
    # won't be pickable -- it should be started/opened from within the new process (of a ProcessRunner). Thus,
    # it's necessary to start the grabber, determine the frame size, close it, and start again with the
    # init_unpickable set to False
    print('Getting the size of the frame: ', end='')
    if source == 'file':
        grabber = FrameGrabberCV2Camera(vid_filename)
#        grabber = imp.get_grabber(source='file', file_device=vid_filename, init_unpickable=True)
    else:
        grabber = imp.get_grabber(source='camera', file_device=0, init_unpickable=True)
    if grabber is None or not grabber.is_opened:
        print("\nCouldn't open the grabber to determine the frame size: exiting")
        exit(1)
    im_shape = grabber.capture()[0][1].shape    # (480, 480, 3)
    print('im_shape : {}'.format(im_shape))
    grabber.close()

    # Define data types
    timestamp_type = ctypes.c_float
    array_type = ctypes.c_uint16

    # Initialize the shared image data structure
    data_structure_type = 1
    if data_structure_type == 0:
        print('Using SharedSingleFrame')
        shared_d = imp.SharedSingleFrame(im_shape=im_shape,
                                         array_type=array_type,
                                         timestamp_type=timestamp_type,
                                         lock=False)
    elif data_structure_type == 1:
        print('Using SharedFrameList')
        shared_d = imp.SharedFrameList(im_shape=im_shape,
                                       nelem=100,
                                       array_type=array_type,
                                       timestamp_type=timestamp_type,
                                       lock=False)
    else:
        raise Exception("data_structure_type {} isn't defined".format(data_structure_type))

    # Create shared_events (can also be assigned from FrameGrabber after its initialization)
    shared_events = imp.SharedEvents()

    # Framegrabber scheduler
    fg_scheduler = imp.Scheduler(dt=0.05)

    # Create the frame grabber object, but don't
    if source == 'file':
        grabber = FrameGrabberCV2Camera(vid_filename)
#        grabber = imp.get_grabber(source='file', file_device=vid_filename, init_unpickable=False)
    else:
        grabber = imp.get_grabber(source='camera', file_device=0, init_unpickable=False)

    # Start the framegrabber
    frame_grabber = imp.FrameGrabberRunner(shared_data=shared_d,
                                           grabber=grabber,
                                           shared_events=shared_events,
                                           scheduler=fg_scheduler)

    # Start a display
    display = imp_SimpleDisplay(shared_data=shared_d,
                                shared_events=shared_events)

    # Start the recorder
    vfpath_split = os.path.splitext(vid_filename)
    vid_rec_filename = vfpath_split[0] + '_rec' + vfpath_split[1]
    recorder = imp_Recorder(shared_data=shared_d,
                            shared_events=shared_events,
                            filename=vid_rec_filename,
                            fourcc='XVID',
                            is_color=True,
                            im_shape=im_shape,
                            fps=30)

    # Start the analysis
    analysis = imp_Analysis(shared_data=shared_d,
                            shared_events=shared_events,
                            vid_av_filename=vid_av_filename)
#    analysis = imp.Consumer()   # emtpy consumer process, complying with generic interprocess commands

    # Start the processes
    print('Starting the processes')
    recorder.start()
    analysis.start()
    display.start()
    # time.sleep(1)
    frame_grabber.start()

    # add all the processes that have to be stopped at the end to the shared_events structure
    for obj in [frame_grabber, display, recorder, analysis]:
        shared_events.add_mp_object(obj)

    # Stop all the processes after t2run seconds:
    t2run = 5
    time.sleep(t2run)
    print('{} sec have expired. Exiting.'.format(t2run))
    shared_events.exitall()

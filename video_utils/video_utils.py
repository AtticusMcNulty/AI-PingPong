import cv2


class VideoUtils:
    def __init__(self):
        pass

    # store video frames
    def read_video(self, video):
        # open video using cv2
        video = cv2.VideoCapture(video)

        frames = []
        success = True

        # loop over frames
        while success:
            # read() returns two things: if frame could be extracted, the frame itself
            success, frame = video.read()

            # store frame if successfully extracted
            if success:
                frames.append(frame)

        return frames

    # write video frames to file
    def write_video(self, videoFrames):
        # get 4-char code used to compress frames
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # get dimensions of frame 1 of frames array
        # returns (width, height, channels)
        width, height, _ = videoFrames[0].shape

        # init VideoWriter object to write the video frames to
        # params: (output filepath, codec, framerate, framesize)
        video_writer = cv2.VideoWriter(
            "output_videos/video.mp4", fourcc, 24, (height, width)
        )

        # write each frame to the video file
        for frame in videoFrames:
            video_writer.write(frame)

        # release the VideoWriter object to close the video file
        video_writer.release()

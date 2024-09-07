# from folder import exposed class
from object_tracker import ObjectTracker
from video_utils import VideoUtils
from team_assigner import TeamAssigner

# facilitates operations on large arrays of data
import numpy as np
import cv2


def assign_players_to_teams(frames, tracks):
    # init TeamAssigner instance
    teamAssigner = TeamAssigner()

    # assign team colors
    teamAssigner.assign_team_colors(frames, tracks)

    # for each frame
    for frameNum, players in enumerate(tracks["player"]):

        playerNum = 0

        # for each player in the frame
        for playerId, player in players.items():
            # assign player to a team based on their color
            team = teamAssigner.get_player_team(
                frames[frameNum], player["bbox"], playerId
            )

            # update player to have a team and team color field
            player["team"] = team
            player["teamColor"] = teamAssigner.teamColors[team]


def main():
    # init VideoUtils instance
    videoUtils = VideoUtils()

    # get list of frames from video
    frames = np.array(videoUtils.read_video("input_videos/video.mp4"))

    # init ObjectTracker instance
    objTracker = ObjectTracker()

    """
    tracks = objTracker.detect_frames(frames)

    videoUtils.write_video([track.plot() for track in tracks])
    return
    """

    # track bboxes and label players
    tracks = objTracker.track_frames(
        frames, read_stub=True, stub_path="stubs/camera_movement_stub.pkl"
    )

    # save cropped image of a player
    for trackId, player in tracks["player"][0].items():
        bbox = player["bbox"]
        frame = frames[0]

        # crop bbox from frame
        croppedImg = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]

        # save the cropped image
        cv2.imwrite(f"output_videos/croppedImg.jpg", croppedImg)

        break

    # assign each player to a team
    assign_players_to_teams(frames, tracks)

    # draw annotations on frames
    annotatedFrames = np.array(objTracker.annotate_frames(frames, tracks))

    videoUtils.write_video(annotatedFrames)


if __name__ == "__main__":
    main()

from ultralytics import YOLO
import sys
import supervision as sv
import pandas as pd
import numpy as np
import pickle
import cv2
import os

sys.path.append("../")


class ObjectTracker:
    def __init__(self):
        self.model = YOLO("models/ping-pong_2.pt")
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        # check objects have already been detected on this video
        # if stub_path is not None and os.path.exists(stub_path)

        # array to store frames with detected objects
        detectedFrames = []
        counter = 1

        # loop through frames
        for frame in frames:
            # use model to predict bounding boxes
            # conf sets min confidence level, objects below 0.15 conf will be disregarded
            # predict returns list of Results objects (in this case just 1)
            detectedFrame = self.model.predict(frame, conf=0.15)[0]

            # extract the image with annotations
            # [0] accesses the single Result object from the list
            # plot returns the image as a numpy array
            detectedFrames.append(detectedFrame)

            counter += 1
            print(f"Detections: {counter}/{len(frames)}")

        return detectedFrames

    def track_frames(self, frames, read_stub=False, stub_path=None):
        if read_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
            return tracks

        # detect objects in frames
        # returns bounding boxes and corresponding class names for each frame
        detectedFrames = self.detect_frames(frames)

        # setup output dict
        tracks = {"racket": [], "net": [], "player": [], "ball": [], "table": []}

        # for each frame
        for frameNum, detectedFrame in enumerate(detectedFrames):

            # format output dict, setup dict for current frame
            tracks["racket"].append({})
            tracks["net"].append({})
            tracks["player"].append({})
            tracks["ball"].append({})
            tracks["table"].append({})

            netBBox = [None, None, None, None]
            tableBBox = [None, None, None, None]

            if detectedFrame.boxes is not None:

                bboxes = detectedFrame.boxes.xyxy.cpu().numpy()
                class_ids = detectedFrame.boxes.cls.cpu().numpy().astype(int)
                confidences = detectedFrame.boxes.conf.cpu().numpy()

                ballConf = 0

                for bbox, class_id, confidence in zip(bboxes, class_ids, confidences):
                    # map class ID to class name
                    className = self.model.names[class_id]

                    # check if the className exists in tracks and initialize if necessary
                    if className not in tracks:
                        print(f"Unexpected class name: {className}")
                        continue

                    ## combine net detections into a single net (top-left, bottom-right)
                    if className == "net":
                        # top-left point
                        if netBBox[0] is None or bbox[0] < netBBox[0]:
                            netBBox[0] = bbox[0]
                        if netBBox[1] is None or bbox[1] < netBBox[1]:
                            netBBox[1] = bbox[1]

                        # bottom-right point
                        if netBBox[2] is None or bbox[2] > netBBox[2]:
                            netBBox[2] = bbox[2]
                        if netBBox[3] is None or bbox[3] > netBBox[3]:
                            netBBox[3] = bbox[3]

                        tracks[className][frameNum][1] = {"bbox": netBBox}
                    ## combine table detections into a single table (top-left, bottom-right)
                    elif className == "table":
                        # top-left point
                        if tableBBox[0] is None or bbox[0] < tableBBox[0]:
                            tableBBox[0] = bbox[0]
                        if tableBBox[1] is None or bbox[1] < tableBBox[1]:
                            tableBBox[1] = bbox[1]

                        # bottom-right point
                        if tableBBox[2] is None or bbox[2] > tableBBox[2]:
                            tableBBox[2] = bbox[2]
                        if tableBBox[3] is None or bbox[3] > tableBBox[3]:
                            tableBBox[3] = bbox[3]

                        tracks[className][frameNum][1] = {"bbox": tableBBox}
                    elif className == "ball":
                        if confidence > ballConf:
                            ballConf = confidence
                            tracks[className][frameNum][1] = {"bbox": bbox}

            # convert current frame to format tracker can use
            supervisionFrame = sv.Detections.from_ultralytics(detectedFrame)

            # add tracker to bboxes of current frame
            # allows us to label players with ids across frames
            trackedFrame = self.tracker.update_with_detections(supervisionFrame)

            for obj in trackedFrame:
                # extract bbox, class_id, and trackId
                bbox = obj[0].tolist()
                trackId = obj[4]
                className = obj[5]["class_name"]

                if className == "player":
                    tracks[className][frameNum][trackId] = {"bbox": bbox}

                elif className == "racket":
                    tracks[className][frameNum][trackId] = {"bbox": bbox}

        # save tracks object in file
        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_square(self, frame, bbox, color):
        cv2.rectangle(
            frame,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            color,
            2,
        )

        return frame

    def draw_triangle(self, frame, bbox, color):
        topY = int(bbox[1])
        centerX = (bbox[0] + bbox[2]) / 2

        # define triangle points
        trianglePoints = np.array(
            [[centerX, topY], [centerX - 10, topY - 20], [centerX + 10, topY - 20]],
            dtype=np.int32,
        )

        cv2.drawContours(frame, [trianglePoints], 0, color, cv2.FILLED)

        return frame

    def draw_id(self, frame, bbox, color, trackId=None):
        xCenter = int((bbox[0] + bbox[2]) / 2)
        topY = int(bbox[1])

        # draw line
        cv2.line(
            frame,
            (xCenter + 20, topY),
            (xCenter - 20, topY),
            color=color,
            thickness=4,
        )

        # if player, draw id
        if trackId is not None:
            cv2.rectangle(
                frame,
                (xCenter - 10, topY - 30),
                (xCenter + 10, topY),
                color,
                cv2.FILLED,
            )

            # draw player id
            cv2.putText(
                frame,
                f"{trackId}",
                (xCenter - 5, topY - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

        return frame

    def interpolate_ball_positions(self, ball_positions):
        # convert ball positions to an array
        ball_positions = [x.get(1, {}).get("bbox", []) for x in ball_positions]

        # convert ball positions to pandas dataframe
        df_ball_positions = pd.DataFrame(
            ball_positions, columns=["x1", "y1", "x2", "y2"]
        )

        # interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        # return ball positions to original format
        ball_positions = [
            {1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()
        ]

        return ball_positions

    def is_ball_in_player_bbox(self, player, ballBBox):
        playerBBox = player["bbox"]

        # get center of ballBBox (x, y)
        ballCenter = [
            (ballBBox[0] + ballBBox[2]) / 2,  # x-coordinate (left + right) / 2
            (ballBBox[1] + ballBBox[3]) / 2,  # y-coordinate (top + bottom) / 2
        ]

        # if center of ball is inside playerBBox
        return (
            ballCenter[0] > playerBBox[0]  # left side
            and ballCenter[0] < playerBBox[2]  # right side
            and ballCenter[1] > playerBBox[1]  # top side
            and ballCenter[1] < playerBBox[3]  # bottom side
        )

    def draw_player_hits(self, frame, playerHits):
        height, width, channels = frame.shape

        # draw a semi-transparent rectangle
        cv2.rectangle(
            frame, (width - 300, height - 150), (width, height), (255, 255, 255), -1
        )
        alpha = 0.4
        cv2.addWeighted(frame, alpha, frame, 1 - alpha, 0, frame)

        # Draw the text for player hits
        cv2.putText(
            frame,
            f"Player 1 Hits: {playerHits[0]}",
            (width - 250, height - 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Player 2 Hits: {playerHits[1]}",
            (width - 250, height - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            2,
        )

        return frame

    def detect_ball_hits(self, frames, ballTracks, tableBBox):
        prevBallY, prevBallY2 = None, None
        updatedFrames = []
        tableHits = 0

        height, width, channels = frames[0].shape

        for frameNum, frame in enumerate(frames):
            if not ballTracks[frameNum] or 1 not in ballTracks[frameNum]:
                cv2.putText(
                    frame,
                    f"Table Hits: {tableHits}",
                    (0, height - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                updatedFrames.append(frame)
                continue

            ballBBox = ballTracks[frameNum][1]["bbox"]
            ballCenter = [
                (ballBBox[0] + ballBBox[2]) / 2,  # x-coordinate
                (ballBBox[1] + ballBBox[3]) / 2,  # y-coordinate
            ]

            # if ball center is within the table's bounding box
            if (
                tableBBox[0] <= ballCenter[0] <= tableBBox[2]
                and tableBBox[1] <= ballCenter[1] <= tableBBox[3]
            ):
                curIncreasing, prevDecreasing = False, False

                # if ball has been detected in frame
                if len(ballTracks[frameNum]) > 0:

                    # if prevBallY2 is None
                    if prevBallY2 == None:
                        # set it to the ball y-val of the current frame
                        prevBallY2 = (ballBBox[1] + ballBBox[3]) / 2

                    # if prevBallY is None
                    elif prevBallY == None:
                        # set it to the ball y-val of the current frame
                        prevBallY = (ballBBox[1] + ballBBox[3]) / 2

                    # if prevBallY and prevBallY2 have been defined
                    else:
                        # get the ball y-val of the current frame
                        curBallY = (ballBBox[1] + ballBBox[3]) / 2

                        tolerance = 1
                        # if current y-val is less than prev y-val, ball is increasing
                        curIncreasing = (curBallY - tolerance) < prevBallY
                        # if prev y-val is greater than prev2 y-val, ball is decreasing
                        prevDecreasing = (prevBallY2 - tolerance) < prevBallY

                        # set new prevs
                        prevBallY2 = prevBallY
                        prevBallY = curBallY

                # if ball is increasing in height after decreasing in height
                if curIncreasing and prevDecreasing:
                    ballTracks[frameNum][1]["ballHit"] = (
                        int(ballCenter[0]),
                        int(ballCenter[1]),
                    )
                    tableHits += 1

                else:
                    ballTracks[frameNum]["ballHit"] = None

            cv2.putText(
                frame,
                f"Table Hits: {tableHits}",
                (0, height - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            updatedFrames.append(frame)

        return updatedFrames

    def annotate_frames(self, frames, tracks):
        annotatedFrames = []

        ## get average table bbox
        netCounter = 0
        tableCounter = 0
        totalNetBBox = np.zeros(4)
        totalTableBBox = np.zeros(4)

        # for each frame
        for frameNum, frame in enumerate(frames):
            netTracks = tracks["net"][frameNum]
            tableTracks = tracks["table"][frameNum]

            for net in netTracks.values():
                if net["bbox"][0] is not None:
                    totalNetBBox += np.array(net["bbox"])
                    netCounter += 1

            for table in tableTracks.values():
                if table["bbox"][0] is not None:
                    totalTableBBox += np.array(table["bbox"])
                    tableCounter += 1

        averageNetBBox = (totalNetBBox / netCounter).tolist()
        averageTableBBox = (totalTableBBox / tableCounter).tolist()

        playerHits = [0, 0]

        # init previous frame player states
        prevPlayerStates = {1: None, 2: None}
        ballInPlayerStates = {1: False, 2: False}

        # for each frame
        for frameNum, frame in enumerate(frames):
            # extract object dicts corresponding to the current frame
            ballTracks = tracks["ball"][frameNum]
            playerTracks = tracks["player"][frameNum]
            racketTracks = tracks["racket"][frameNum]

            frame = self.draw_square(frame, averageNetBBox, (0, 0, 255))
            frame = self.draw_square(frame, averageTableBBox, (0, 255, 0))

            """
            for racketId, racket in racketTracks.items():
                if racket["bbox"]:
                    frame = self.draw_triangle(frame, racket["bbox"], (255, 0, 0))
            """

            for playerId, player in playerTracks.items():
                if player["bbox"]:
                    frame = self.draw_id(
                        frame, player["bbox"], player["teamColor"], (player["team"] + 1)
                    )

                    # check if player had the ball in the previous frame
                    team = player["team"]
                    prevPlayer = prevPlayerStates[team]

                    # determine if the ball is within the player's bounding box
                    ballInBBox = False
                    for ballId, ball in ballTracks.items():
                        if ball["bbox"] is not None:
                            ballInBBox = self.is_ball_in_player_bbox(
                                player, ball["bbox"]
                            )
                            if ballInBBox and not ballInPlayerStates[team]:
                                # ball just entered the player's bounding box
                                playerHits[team - 1] += 1
                                ballInPlayerStates[team] = True
                            elif not ballInBBox and ballInPlayerStates[team]:
                                # ball just left the player's bounding box
                                ballInPlayerStates[team] = False

                            self.draw_triangle(frame, ball["bbox"], (255, 140, 0))

                # update previous state for the current team
                prevPlayerStates[team] = player

            frame = self.draw_player_hits(frame, playerHits)

            annotatedFrames.append(frame)

        # draw ball hits
        annotatedFrames = self.detect_ball_hits(
            annotatedFrames, tracks["ball"], averageTableBBox
        )

        return annotatedFrames

from view_transformer import ViewTransformer
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
            """
            Use model to predict bounding boxes:
                conf sets min confidence level, objects below 0.15 conf will be disregarded
                predict returns list of Results objects (in this case just 1)
                iou (intersection over union) uses nms (non-maximum suppression) to only keep bounding box with the height confidence score
                    the threshold determines how much overlap is tolerated before removing a bounding box
            """
            detectedFrame = self.model.predict(frame, conf=0.15, iou=0.1)[0]

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

    def detect_table_hits(self, frames, ballTracks, tableBBox):
        prevBallCenters, prevBallBBoxes = [], []
        updatedFrames = []
        yTol = 4
        tableHits = 0
        cooldownCounter = 0
        cooldownTime = 5

        height, width, _ = frames[0].shape

        # set bbox ball size
        fixedBallBBoxSize = 20

        # Determine the fixed ball size by averaging over initial detections
        for ball in ballTracks:
            if ball and 1 in ball:
                fixedBallBBoxSize = ball[1]["bbox"][3] - ball[1]["bbox"][1]
                break

        for frameNum, frame in enumerate(frames):
            # decrease the cooldown counter if it's active
            if cooldownCounter > 0:
                cooldownCounter -= 1

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

            # get ball bbox size, and normalize center
            ballBBox = ballTracks[frameNum][1]["bbox"]
            ballCenter = [
                (ballBBox[0] + fixedBallBBoxSize / 2),
                (ballBBox[1] + fixedBallBBoxSize / 2),
            ]

            cv2.circle(
                frame,
                (int(ballCenter[0]), int(ballCenter[1])),
                3,
                (0, 255, 0),
                2,
            )

            # append current ball position to the list
            prevBallCenters.append(ballCenter)
            prevBallBBoxes.append(ballBBox)

            # smooth the ball bbox size over time (moving average)
            if len(prevBallBBoxes) > 5:
                fixedBallBBoxSize = np.mean(
                    [bbox[3] - bbox[1] for bbox in prevBallBBoxes[-5:]]
                )

            # keep only the last 3 positions (or adjust the number based on testing)
            if len(prevBallCenters) > 3:
                prevBallCenters.pop(0)

            # ensure there are enough positions to check
            if len(prevBallCenters) == 3:
                # (prev - prev2) > 0
                descending = (prevBallCenters[1][1] - prevBallCenters[0][1] + yTol) > 0
                # (prev - cur) > 0
                ascending = (prevBallCenters[1][1] - prevBallCenters[2][1] + yTol) > 0

                # if the ball was descending and then ascending
                if descending and ascending and not cooldownCounter > 0:
                    # if the ball is within the table's bounding box
                    if (
                        tableBBox[0] <= prevBallCenters[1][0] <= tableBBox[2]
                        and tableBBox[1] <= prevBallCenters[1][1] <= tableBBox[3]
                    ):
                        ballTracks[frameNum][1]["ballHit"] = [
                            int(ballCenter[0]),
                            int(ballCenter[1]),
                        ]
                        tableHits += 1
                        cooldownCounter = cooldownTime

            # display the number of table hits on the frame
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

        # draw table hits
        frames = self.detect_table_hits(frames, tracks["ball"], averageTableBBox)

        # define vars to store:
        # player hits
        # previous ball in player states
        # ball hit positions across frames
        playerHits = [0, 0]
        ballInPlayerStates = {1: False, 2: False}
        ballHitPositions = []

        # init view transformer
        viewTransformer = ViewTransformer()

        # for each frame
        for frameNum, frame in enumerate(frames):
            # extract object dicts corresponding to the current frame
            ballTracks = tracks["ball"][frameNum]
            playerTracks = tracks["player"][frameNum]

            frame = self.draw_square(frame, averageNetBBox, (0, 0, 255))
            frame = self.draw_square(frame, averageTableBBox, (0, 255, 0))

            for playerId, player in playerTracks.items():
                if player["bbox"]:
                    frame = self.draw_id(
                        frame, player["bbox"], player["teamColor"], (player["team"] + 1)
                    )

                    # check if player had the ball in the previous frame
                    team = player["team"]

                    # determine if the ball is within the player's bounding box
                    ballInBBox = False

                    # if ball exists in cur frame
                    if len(ballTracks) > 0:
                        # check if ball is in player bbox
                        ballInBBox = self.is_ball_in_player_bbox(
                            player, ballTracks[1]["bbox"]
                        )

                        # if ball is in player bbox and not in it previously
                        if ballInBBox and not ballInPlayerStates[team]:
                            # ball just entered the player's bounding box
                            playerHits[team - 1] += 1
                            ballInPlayerStates[team] = True
                        elif not ballInBBox and ballInPlayerStates[team]:
                            # ball just left the player's bounding box
                            ballInPlayerStates[team] = False

                        self.draw_triangle(frame, ballTracks[1]["bbox"], (255, 140, 0))

                        # track ball hit positions
                        if "ballHit" in ballTracks[1]:
                            ballHitPositions.append(ballTracks[1]["ballHit"])

            frame = self.draw_player_hits(frame, playerHits)

            # draw ball hits on the transformed table view
            tableHitMap = viewTransformer.transform_frames(
                frames[len(frames) - 1], ballHitPositions, averageTableBBox
            )

            # resize the table view while maintaining aspect ratio
            height, width = frame.shape[:2]
            aspectRatio = width / height
            newWidth = 150
            newHeight = int(newWidth / aspectRatio)
            tableHitMapSmall = cv2.resize(tableHitMap, (newWidth, newHeight))

            # overlay the resized table view in the corner of the frame
            frame[0:newHeight, width - newWidth : width] = tableHitMapSmall

            annotatedFrames.append(frame)

        return annotatedFrames

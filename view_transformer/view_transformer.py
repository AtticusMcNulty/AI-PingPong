import numpy as np
import cv2


class ViewTransformer:
    def __init__(self):
        pass

    def transform_table_view(self, frame, tableBBox):
        # define source points from the table's bounding box (corners)
        sourcePoints = np.float32(
            [
                [tableBBox[0], tableBBox[1]],
                [tableBBox[2], tableBBox[1]],
                [tableBBox[2], tableBBox[3]],
                [tableBBox[0], tableBBox[3]],
            ]
        )

        # define destination points for a top-down view (a square of 500x500 pixels)
        destPoints = np.float32(
            [
                [0, 0],
                [500, 0],
                [500, 500],
                [0, 500],
            ]
        )

        # calculate the perspective transform matrix
        transformMatrix = cv2.getPerspectiveTransform(sourcePoints, destPoints)

        # apply the perspective transformation to get a top-down view
        topDownView = cv2.warpPerspective(frame, transformMatrix, (500, 500))

        return transformMatrix, topDownView

    def transform_frames(self, lastFrame, ballHitPositions, tableBBox):
        # prepare a blank table hit map
        tableHitMap = np.zeros((500, 500, 3), dtype=np.uint8)

        # transform the table view to a top-down view
        transformMatrix, topDownView = self.transform_table_view(lastFrame, tableBBox)

        # if ballTracks exists and ballHit coords exist
        for ballHit in ballHitPositions:
            # format ball center
            ballCenter = np.array(ballHit).reshape(-1, 1, 2).astype(np.float32)

            # mark the ball hit positions on the transformed table view
            ballCenterTransformed = cv2.perspectiveTransform(
                ballCenter, transformMatrix
            )

            # after processing all frames, mark all ball hit positions on the table hit map
            x, y = int(ballCenterTransformed[0][0][0]), int(
                ballCenterTransformed[0][0][1]
            )

            cv2.circle(tableHitMap, (x, y), radius=5, color=(0, 0, 255), thickness=-1)

        # after processing all frames, overlay the accumulated hits on the top-down view of the last frame
        finalView = cv2.addWeighted(topDownView, 1, tableHitMap, 1, 0)

        return finalView

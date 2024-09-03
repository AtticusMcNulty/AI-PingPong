from sklearn.cluster import KMeans


class TeamAssigner:
    def __init__(self):
        # dict to store colors for each team
        self.teamColors = {}

        # dict to store team assignment for each player
        self.playerTeamDict = {}

    def get_clustering_model(self, img):
        # reshape image into 2d array of pixels
        img2D = img.reshape(-1, 3)

        # preform KMeans clustering with 2 clusters (background and player colors)
        kmeans = KMeans(n_clusters=4, init="k-means++", n_init=1)
        kmeans.fit(img2D)

        # return the kmeans model
        return kmeans

    def get_player_color(self, frame, bbox):
        # crop frame around player
        croppedImg = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]

        # get top half of cropped image
        topHalfImg = croppedImg[0 : int(croppedImg.shape[0] / 2), :]

        # get the KMeans clustering model for the top half of the image
        kmeans = self.get_clustering_model(topHalfImg)

        # Get the cluster labels and centers (dominant colors)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        # reshape the labels to the original image shape
        clusteredImage = labels.reshape(topHalfImg.shape[0], topHalfImg.shape[1])

        # map each cluster label to its corresponding color
        clusteredImageRGB = centers[clusteredImage].astype(int)

        height, width, _ = clusteredImageRGB.shape
        centerX, centerY = width // 2, height // 2
        centerPixelRGB = clusteredImageRGB[centerY, centerX]

        return centerPixelRGB

    def assign_team_colors(self, frames, tracks):
        playerColors = []
        player1BBox, player2BBox = None, None

        # for each frame
        for frameNum, frame in enumerate(frames):
            # print(f"Players at frame {frameNum}: {tracks['player'][frameNum]}")

            # if the number of players equals 2 (able to get 2 teams)
            if len(tracks["player"][frameNum]) == 2:

                # get bbox of each player
                player1BBox = tracks["player"][frameNum][1]["bbox"]
                player2BBox = tracks["player"][frameNum][2]["bbox"]

                # append color of both players to playerColors
                playerColors.append(self.get_player_color(frame, player1BBox))
                playerColors.append(self.get_player_color(frame, player2BBox))

                break
            else:
                continue

        # use KMeans to cluster player colors into 2 teams
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(playerColors)

        # store the KMeans model for later use
        self.kmeans = kmeans

        # assign the cluster centers as team colors
        self.teamColors[1] = kmeans.cluster_centers_[0]
        self.teamColors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, playerId):
        # check if the player has already been assigned a team
        if playerId in self.playerTeamDict:
            return self.playerTeamDict[playerId]

        # get the player's color from the frame
        player_color = self.get_player_color(frame, player_bbox)

        # predict the team ID based on the player's color
        teamId = self.kmeans.predict(player_color.reshape(1, -1))[0]
        # convert from 0-indexed to 1-indexed
        teamId += 1

        # store the player's team assignment
        self.playerTeamDict[playerId] = teamId

        # return the team id
        return teamId

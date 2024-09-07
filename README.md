# AI Ping Pong
## How I got this Idea
I developed this project after my sophomore school year. Going into the summertime, I wanted to look more into how to actually use machine learning to create a project. At the time both my dad and I were really into playing ping pong, and that gave me the idea to create something that would track a game of ping pong between us and provide some cool information.

## Learning behind the Project
Because I had no prior experience with machine learning, I had to do a lot of research throughout and even before starting this project. I found a couple videos on youtube to be especially helpful as well as the documentation for many of the used libraries. Resources, all of which I have linked below.

## Project Description
This project involved training a YOLO model on a dataset designed for ping pong detection. The trained model is used to track several key elements in a video of a ping pong game. Specifically, the model generates bounding boxes that allow us to highlight the table, the net, the players, and the ball. With these bounding boxes, I used KMeans to cluster players into teams based on their t-shirt colors. Additionally, the project tracks interactions between players and the ball, recording each instance where the ball is hit. To provide more analysis, I applied perspective transformation techniques using OpenCV (cv2). This transformation maps the locations where the ball hits the table, allowing for a visual representation of these points throughout the entire video.

## Video Showcase

## Resources
### Videos/Websites
Extracting BBoxes with YOLO: https://www.youtube.com/watch?v=QtsI0TnwDZs</br>
Using KMeans clustering:</br>
  https://www.w3schools.com/python/python_ml_k-means.asp</br>
  https://cierra-andaur.medium.com/using-k-means-clustering-for-image-segmentation-fe86c3b39bf4</br>

### Documentation
CV2: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html</br>
Ultralytics/Yolo: https://docs.ultralytics.com/</br>
Supervision: https://supervision.roboflow.com/latest/detection/core/</br>
KMeans: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html</br>

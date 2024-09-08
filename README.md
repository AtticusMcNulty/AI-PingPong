# AI Ping Pong
## How I got this Idea
I developed this project after my sophomore school year. Going into the summertime, I wanted to look more into how to actually use machine learning in a program. At the time both my dad and I were really into playing ping pong, and that gave me the idea to create something that would track a game between us. With the end goal of providing interesting and relevant information about the match.

## Learning behind the Project
Because I had no prior experience with machine learning, I had to do a lot of research throughout and even before starting this project. I found a couple online videos and articles to be especially helpful as well as the documentation for many of the used libraries (all of which I have linked below).

## Project Description
This project involved training an object detection model from YOLO on a ping pong dataset I got from roboflow. The training for this model was done using the free version of Google Colab, as they provide GPU usage for a couple hours each day.</br>
The finalized model was then used to track several key elements in a video of a ping pong game. The program begins by generating and storing the bounding boxes of key elements (table, net, players, ball, etc.) in each frame. With these bounding boxes, I use the KMeans algorithm to cluster players into teams based on their shirt colors. This allows me to preform more complex calculations such as separate counters that record each time a player hits the ball.</br> 
Most recently, I implemeneted perspective transformation using OpenCV (cv2). This enabled me to map the locations where the ball hits the table and provide visual representation of these points throughout the entire video.

## Visual Showcase
The program begins by reading an input video and storing each of its frames in an array.</br>
<img width="523" alt="Screenshot 2024-09-07 at 6 41 02 PM" src="https://github.com/user-attachments/assets/c540e9e5-3c16-4391-be74-716a7b23089d">

We then call the main detection function of our program "track_frames".</br>
<img width="513" alt="Screenshot 2024-09-07 at 6 42 57 PM" src="https://github.com/user-attachments/assets/0e478837-4b55-4ec2-962b-43a2fed48c7b"></br>
From track_frames we call detect_frames which runs our model on each of the frames and stores the result.</br>
<img width="514" alt="Screenshot 2024-09-07 at 6 48 48 PM" src="https://github.com/user-attachments/assets/53095da6-fea6-4151-9212-e2a5ab8a366b"></br>
We then loop through the stored frames and format the results into an object. For table/net detections, we combine their detections to get the entire table/net (table detects with other tables, nets with other nets).</br>
<img width="532" alt="Screenshot 2024-09-07 at 6 59 37 PM" src="https://github.com/user-attachments/assets/d952afbd-d923-43f0-a551-f1dd584b2476"></br>
For the ball, we simply store the ball detection each frame with the highest confidence.</br> 
<img width="452" alt="Screenshot 2024-09-07 at 7 00 31 PM" src="https://github.com/user-attachments/assets/adc0ddfd-ece7-41b2-9ad2-853d4f2554a7"></br>
<img width="411" alt="Screenshot 2024-09-07 at 7 00 45 PM" src="https://github.com/user-attachments/assets/bbc03c79-ada9-4382-a351-d5416c5a7afc"></br>
Finally, for players we apply a tracker using supervision which allows us to track the two players across the video.</br>
<img width="503" alt="Screenshot 2024-09-07 at 7 01 10 PM" src="https://github.com/user-attachments/assets/bd928343-d95e-4e67-aeb4-79382451c0ad"></br>

## Resources
### Videos/Websites
Extracting BBoxes with YOLO: https://www.youtube.com/watch?v=QtsI0TnwDZs</br>
KMeans Image clustering:</br>
  https://www.w3schools.com/python/python_ml_k-means.asp</br>
  https://cierra-andaur.medium.com/using-k-means-clustering-for-image-segmentation-fe86c3b39bf4</br>
Training Models with Google Colab: https://colab.research.google.com/github/tensorflow/swift/blob/main/docs/site/tutorials/model_training_walkthrough.ipynb</br>

### Documentation
CV2: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html</br>
Ultralytics/Yolo: https://docs.ultralytics.com/</br>
Supervision: https://supervision.roboflow.com/latest/detection/core/</br>
KMeans: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html</br>

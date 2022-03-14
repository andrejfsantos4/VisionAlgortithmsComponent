# VisionAlgortithmsComponent
This repository offers a gRPC service for building pipelines that executes several Computer Vision tasks and algorithms, such as feature detection and matching, homography 
estimation, object tracking and more. 

It was developed according to the specifications of [AI4Europe](https://www.ai4europe.eu/), the European AI-on-demand platform, and will soon be available on the 
[Acumos](https://www.acumos.org/) platform. 

## Deployment
Although it can be deployed as a standalone gRPC server that waits for requests from clients, more complex workflows can be achieved using and orchestrator such as 
[Pipeline-Orchestrator](https://github.com/DuarteMRAlves/Pipeline-Orchestrator). In the latter case, this component can be pulled from DockerHub

```docker pull andrejfsantos/vision-algorithms:latest```

## Algorithms
Currently, the following tasks are supported:
- SIFT features detection in an image
- SIFT features matching between two sets of features descriptors
- Homography estimation either with initial least-squares method and Levenberg-Marquardt method for refinement. In the presence of outliers, a robust method such as 
RANSAC is also available
- Warping of an image or set of points with a user-defined homography

The following tasks are currently in development:
- Object tracking with Kanade-Lukas-Tomasi (KLT) tracker
- Line detection with Hough transform

## Inputs and Outputs
[vision_algorithms.proto](protos/vision_algorithms.proto) defines the interface of this service. It receives an ExecRequest which specifies the task to execute and its arguments.
Each task has its own message type which includes all necessary and optional arguments. Likewise, the service outputs an ExecResponse which will contain a message with the 
output data. [parsing.py](parsing.py) contains several methods for manipulating data between structures such as Numpy arrays, and Protobuf messages.

Some examples of how to use this service can be seen in this [client](vision_algorithms_client.py). 

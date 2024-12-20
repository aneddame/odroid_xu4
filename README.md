![Time Spent Graph](time_spent_graph.png)

YOLOv4: This method took the most time, with a total of 34.9722 seconds. YOLOv4 (You Only Look Once version 4) is a popular and powerful deep learning model for real-time object detection. However, its processing time can be higher due to the complexity of the model and the need for high computational resources.

OpenCV DNN: The second method, OpenCV DNN, took significantly less time, with a total of 1.2876 seconds. OpenCV's deep neural network (DNN) module allows for running pre-trained models, and it is optimized for efficiency. The time here is much lower compared to YOLOv4, indicating that the OpenCV DNN approach is faster, likely due to lighter models or optimizations.

OpenCV DNN with OpenMP: The third method, OpenCV DNN with OpenMP, took 1.25977 seconds, which is almost the same as the previous method but slightly faster. OpenMP (Open Multi-Processing) is a parallel programming model that allows for better utilization of multi-core processors, leading to improved performance.

Key Insights:
YOLOv4 is much slower than OpenCV DNN and OpenCV DNN with OpenMP. This suggests that while YOLOv4 is highly accurate, it is computationally intensive.
OpenCV DNN and OpenCV DNN with OpenMP are significantly faster, with the OpenMP-optimized version being marginally quicker.
This comparison highlights the trade-offs between accuracy (YOLOv4) and speed (OpenCV DNN, OpenCV DNN with OpenMP), which is a common consideration when choosing a model for real-time applications.

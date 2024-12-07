+++
title = "KhaoManee : face-detection and face-recognition API for CPU"
date = "2021-01-09"
github = "https://github.com/RTae/KhaoManee"
+++

A API provides face bounding box and face-landmarks for face-detection. this project use face-detection model as MTCNN model that returns 5 points face-landmark and Facenet fro face-recognition model.

<!--more-->

[GitHub Repository](https://github.com/RTae/KhaoManee)

KhaoManee uses an embedding vector with a 512-dimension to find a similarity between people. This model can recognize a few members of Parliament in Thailand, such as **Prayut Chan-o-cha**, **Prawit Wongsuwan**, and **Thanathorn Juangroongruangkit**. Also, this project implements the model with Pytorch and the API with FastAPI.

![Alt Text](https://raw.githubusercontent.com/RTae/KhaoManee/main/example/result.png "Example")
# Pneumonia Detection from Chest X Rays

This project uses a convolutional neural network built with TensorFlow and Keras to look at a chest X ray and decide whether it shows signs of pneumonia or looks normal. It comes with a FastAPI backend that serves predictions and a React frontend where you can upload an image and see the result right away.

## About the Dataset

Since the dataset is large, it is not included in this repository. Before running the project, you will need to create the data folder yourself and download the images from Kaggle.

Head over to the Chest X Ray Images Pneumonia page on Kaggle and download the zip file from there. The link is https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Once downloaded, extract the contents and place them inside a folder named data in the root of this project. Inside that folder you should end up with three subfolders named train, val and test, and each of those should contain two folders called NORMAL and PNEUMONIA with the actual X ray images in them.

After this is done, your project folder should look complete and ready to run. The scripts in this repository expect the data to live in exactly this structure, so getting this part right before moving on will save you from confusing errors later.

## Cloning the Project

Start by cloning the repository to your own machine.

```
git clone https://github.com/kgayanjith/pneumonia-detection-deep-learning.git
cd pneumonia-detection-deep-learning
```

## Setting Up the Python Environment

It is best to work inside a virtual environment so your packages stay isolated from the rest of your system.

```
python -m venv venv
```

On macOS or Linux, activate it with

```
source venv/bin/activate
```

On Windows, activate it with

```
venv\Scripts\activate
```

Once the environment is active, install everything the project needs.

```
pip install -r requirements.txt
```

If you are working on an Apple Silicon Mac and want to use the GPU during training, also install the Metal plugin.

```
pip install tensorflow-metal
```

## Checking Your Setup

Before training anything, it helps to confirm that TensorFlow is installed correctly and that your data folder is where it should be.

```
python check_setup.py
```

This will print the TensorFlow version, tell you whether a GPU was detected, and list how many images are found inside each class folder. If you see a file not found error here, it almost always means the data folder has not been placed correctly.

## Training the Model

Once the setup check passes, you can train the model.

```
python train.py
```

This script merges the train and validation folders, creates a fresh and properly balanced split, applies class weighting so the model does not become biased toward the larger class, and then trains the convolutional neural network. When it finishes, it saves the trained model as pneumonia_model.keras in your project folder.

## Evaluating the Model

After training, you can test how well the model performs on data it has never seen before.

```
python evaluate.py
```

This will print precision, recall, F1 score and the confusion matrix, along with the ROC AUC score. Pay close attention to recall for the pneumonia class, since missing a real case matters far more than a false alarm.

## Running the Backend API

With a trained model saved, you can start the FastAPI server that exposes a prediction endpoint.

```
uvicorn api:app --reload
```

This will start the server at http://127.0.0.1:8000. Keep this running while you use the frontend.

## Running the Frontend

Open a new terminal window, move into the frontend folder, install its dependencies, and start the development server.

```
cd pneumonia-frontend
npm install
npm run dev
```

Once it is running, open your browser and go to http://localhost:5173. From there you can upload a chest X ray and see whether the model predicts normal or pneumonia, along with a confidence score.

## A Quick Note Before You Start

Make sure the backend server from the previous step is still running while you use the frontend, since the website sends the uploaded image to that server to get a prediction back.

## Disclaimer

This project was built for learning purposes as part of an academic assignment. It is not a medical tool and should never be used to make real diagnostic decisions. Always consult a qualified doctor for anything related to actual health concerns.

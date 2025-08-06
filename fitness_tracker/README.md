# Tracking Barbell Exercises
*Master Project | Abhijeet Thombare | IIT, Guwahati- DS AI, India*
<p align="right"> abhithombare45 </p>
This repository provides all the code to process, visualize, and classify accelerometer and gyroscope data obtained from fitbit smart band. The data was collected during gym workouts where participants were performing various barbell exercises.

#### Exercises
![Barbell exercise examples](images/barbell_exercises.png)
![Barbell exercise graphs](images/graphs.png)

#### Goals
* Classify barbell exercises
* Count repetitions
* Detect improper form 

#### Installation
Create and activate an anaconda environment and install all package versions using `conda install --name <EnvironmentName> --file conda_requirements.txt`. Install non-conda packages using pip: `pip install -r pip_requirements.txt`.

#### References
The original code is associated with the book titled "*Machine Learning for the Quantified Self: On the Art of Learning from Sensory Data*"
authored by Mark Hoogendoorn and Burkhardt Funk and published by Springer in 2017. The website of the book can be found on [ml4qs.org](https://ml4qs.org/).

#### Project Structure 
```
├── LICENSE
├── README.md         # Project overview and usage guide
├── data              # Data storage directory
│   ├── external      # Data from third-party sources
│   ├── interim       # Intermediate transformed datasets
│   ├── processed     # Final, clean datasets for modeling
│   └── raw           # Original sensor data
├── docs              # Documentation files
├── models            # Trained models and predictions
├── notebooks         # Jupyter notebooks for analysis
├── references        # Manuals and explanatory materials
├── reports           # Analysis reports with figures
├── src               # Source code for project
│   ├── data          # Data preparation scripts
│   ├── features      # Feature engineering scripts
│   ├── models        # Model training and evaluation scripts
│   └── visualization # Scripts for data visualization
└── requirements.txt  # Dependencies for project setup
```
#### Acknowledgments
This project is inspired by the research conducted by Dave Ebbelaar. His paper, "Exploring the Possibilities of Context-Aware Applications for Strength Training", provided the foundation for the methodologies and approaches used in this project.

######  Technologies Used

####   Hardware
  + Meta Motion Sensor (Wrist-Worn): Captures accelerometer and gyroscope data.

####   Software

  - Programming Language: Python
  - Libraries:
      * Data Processing: pandas, numpy, scipy
      * Machine Learning: scikit-learn
      * Visualization: matplotlib, seaborn
  - Development Tools:
  - Jupyter Notebooks for exploration and experimentation.
  - Pickle for storing and loading intermediate datasets.

######  How It Works

####  1. Data Collection
The dataset is collected using Meta Motion sensors worn on the wrist, mimicking the placement of a smartwatch.
The sensors capture accelerometer (measuring acceleration along X, Y, Z axes) and gyroscope (measuring angular velocity) data during strength training exercises.
Five foundational barbell exercises are recorded:
Bench Press
Deadlift
Overhead Press
Barbell Row
Squat Alt Text

####  2. Data Processing
Outlier Removal: Detected and removed anomalies from the dataset using Local Outlier Factor (LOF).
Data Smoothing: Applied low-pass filtering to remove high-frequency noise and ensure smooth signals.
Temporal Aggregation: Summarized motion data over time windows, calculating mean, standard deviation, and other metrics.
Frequency Analysis: Used Fourier Transformations to identify periodic patterns, such as repetitions in exercises.

####  3. Feature Engineering
Dimensionality Reduction: Used Principal Component Analysis (PCA) to simplify the dataset while retaining critical motion patterns.
Clustering-Based Features: Grouped similar movements using k-means clustering to distinguish between overlapping exercises.
Scalar Magnitudes: Combined sensor data into orientation-independent metrics to capture overall motion intensity.

####  4. Model Training
Machine learning models are trained to classify exercises, count repetitions, and detect improper form:
Classification Accuracy: Achieved 98.5% using Random Forest for exercise identification.
Repetition Counting: Used peak detection algorithms to count repetitions with ~95% accuracy.
Form Detection: Trained separate models to detect improper execution of exercises (e.g., incorrect bench press form).

####  5. Results
Exercise Classification Accuracy: 98.5%
Repetition Counting Error: ~5%
Form Detection Accuracy: 98.5%

> Project: Tracking Barbell Exercises using ML on Sensor Data [ Master Project | Abhijeet Thombare | IIT Guwahati Feb 2025 ]  
> Built a fitness tracker model using time-series sensor data to classify different barbell exercises. Includes preprocessing, feature extraction, and supervised learning models.  


<p align="right">— Abhijeet Thombare  </p>

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

#### Project Structure ```
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


> Project: Tracking Barbell Exercises using ML on Sensor Data [ Master Project | Abhijeet Thombare | IIT Guwahati Feb 2025 ]  
> Built a fitness tracker model using time-series sensor data to classify different barbell exercises. Includes preprocessing, feature extraction, and supervised learning models.  


<p align="right">— Abhijeet Thombare  </p>

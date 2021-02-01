# :school: College-Enquiry-Chatbot
<p align="center">
  <img src = src/bot.png >
</p>

# Table of Contents
1. [About The Project](#about-the-project)
    * [Motivation](#motivation)
    * [Flowchart](#flowchart)
    * [Methodology](#methodology)
2. [Getting Started](#getting-started)   
    * [Prerequisites](#prerequisites)
    * [Installation](#installation)  
3. [Contributors](#contributors)

# About the project
The College Enquiry Chatbot is built to help the students resolve all their queries regarding college ***infrastructure, credits, curriculum and other facilities***. The bot **UI** has cool font styles and background thereby creating a personalized branded experience. It communicates with students on all aspects of college life, creating a virtual "one-stop-shop" for :student: queries. 
<h4>Mode of Communication: text.</h4>

An example of the working bot would be something like this:
```
You: Hi
bot: How can I help you?
You: Tell about ffcs
bot: With ffcs, a student can prepare his/her own timetable with the specific courses he/she intends to do in that semester along with the timings of classes and choice of professors. 
```

## Motivation
Freshers have many doubts regarding the ***Credit System, extracurricular activities*** and many of them don't know whom to contact for their issues :confused:. So what this means is that at the beginning of every academic year, the university’s faculty and staff members are burdened with the additional responsibility of showing new students the ropes and answering the same bunch of questions. This chatbot solves all these problems and reduce burden on faculties by giving an instant and accurate response.

## Flowchart
<p align="center">
  <img src = src/flowchart.png width="650" height="650">
</p>

## Methodology
Our college enquiry chatbot is developed using python that analyses user’s queries and responds in a friendly manner. We have created our own intents.json file.  The data file which has predefined patterns and responses.
- Import and load the data file – 
    * We import the necessary packages for our chatbot and initialize the variables we will use in our Python project.
-	Preprocess data – 
    * When working with text data, we need to perform various preprocessing on the data before we make a machine learning or a deep learning model. Stop word removal, tokenization, etc. come in this phase.
- Create the training data –
    * The words and their respective tags must be matched. These are stored is lists named train and output.
-	Build the model –
    * We have our training data ready, now we will build a deep neural network that has 3 layers. We use the keras sequential API for this. We are training the model for 200 epochs and then save it.
-	Predict the response –
    * We will load the trained model and then use a graphical user interface that will predict the response from the bot. The model will only tell us the class it belongs to, so we will implement some functions which will identify the class and then retrieve us a random response from the list of responses. 

# Getting Started
Download the ***src*** folder and copy it in your local machine.

## Prerequisites
For Windows users, it is strongly recommended that you go through this guide to install [Python 3](https://docs.python-guide.org/starting/install3/win/#install3-windows) successfully.
## Installation
1. Install [NLTK](http://pypi.python.org/pypi/nltk).
NLTK requires Python versions 3.5, 3.6, 3.7, or 3.8
2. Install [Numpy](https://www.scipy.org/scipylib/download.html) (optional) 
3. Testing:
```
Start > Python38
Type import nltk
```
4. Installing NLTK data:
```
Import nltk.
nltk.download(‘popular’)
```
5. Run each training model and save it.`py training.py`
6. Open the chat.py and paste the location of the saved model in `model=tf.keras.models.load_model("your directory location").`

# Contributors
* [Ashwin S Guptha](https://github.com/AshwinGuptha)
* [Madhurima Magesh](https://www.linkedin.com/in/madhurima-magesh-586a561a5/)



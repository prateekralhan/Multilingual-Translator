# ✨ Multilingual Translator 💬🗣  [![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active) [![](https://img.shields.io/badge/Prateek-Ralhan-brightgreen.svg?colorB=ff0000)](https://prateekralhan.github.io/)

A simple Streamlit based webapp to translate text to or between numerous languages with mBART-50 from Huggingface and Facebook.

<kbd>
<img src="https://user-images.githubusercontent.com/29462447/149639021-5fc99131-029d-4f21-8715-ebdd679c5263.gif" data-canonical-src="https://user-images.githubusercontent.com/29462447/149639021-5fc99131-029d-4f21-8715-ebdd679c5263.gif"/> 
</kbd>

## Installation:
* Simply run the command ***pip install -r requirements.txt*** to install the dependencies.

## Usage:
1. Clone this repository and install the dependencies as mentioned above.
2. Simply run the command: ***streamlit run app.py***
3. Navigate to http://localhost:8501 in your web-browser.


### Running the Dockerized App
1. Ensure you have Docker Installed and Setup in your OS (Windows/Mac/Linux). For detailed Instructions, please refer [this.](https://docs.docker.com/engine/install/)
2. Navigate to the folder where you have cloned this repository ( where the ***Dockerfile*** is present ).
3. Build the Docker Image (don't forget the dot!! :smile: ): 
```
docker build -f Dockerfile -t app:latest .
```
4. Run the docker:
```
docker run -p 8501:8501 app:latest
```

This will launch the dockerized app. Navigate to ***http://localhost:8501/*** in your browser to have a look at your application. You can check the status of your all available running dockers by:
```
docker ps
```

# APES-on-TAC2011

This repository is meant to access APES summarization evaluation metric on the data published in [TAC AESOP 2011](https://tac.nist.gov/2011/Summarization/AESOP.2011.guidelines.html).

All dependencies can be installed via:
```pip install -r requirements.txt```

## Running steps
### Obtain data
In order to obtain relevant AESOP 2011 data please refer to [AESOP 2011 website](https://tac.nist.gov/2011/Summarization/AESOP.2011.guidelines.html) and submit a request. The data can not be added to this repository.
### Preprocess the data
In order to run the preprocessing script on the data please run the following command:
```python apes_on_tac2011.py --mode preprocess```

### Run QA script
```python apes_on_tac2011.py --mode answer_questions```
This will expect a QA system that reads questions from `./queriesX.pkl` file and writes answering accuracy in `./rewardsX.txt`.
The suffix `X` in the file names represent different run ids.
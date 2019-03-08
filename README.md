# APES-on-TAC2011

This repository is meant to access APES summarization evaluation metric on the data published in [TAC AESOP 2011](https://tac.nist.gov/2011/Summarization/AESOP.2011.guidelines.html).

## Running steps
### Run preprocess
In order to run the preprocessing script on the data please run the following command:
```python apes_on_tac2011.py --mode preprocess```
Three examples from the original AESOP dataset are available in the `examples` folder.

### Obtain data
In order to obtain relevant AESOP 2011 data please refer to [AESOP 2011 website](https://tac.nist.gov/2011/Summarization/AESOP.2011.guidelines.html) and submit a request. The data can not be added to this repository.
### Preprocess the data

### Answer Questions
#### Run QA script
To run the QA you need a QA stream that expects `queries.pkl` file and writes answering accuracy in `rewards.txt`.
You can find a trained QA [here](https://github.com/mataney/rc-cnn-dailymail), so it is required you run the QA stream before running the answering_questions process.

#### QA accuracy

```python apes_on_tac2011.py --mode answer_questions```
This will expect a QA system that reads questions from `./queries.pkl` file and writes answering accuracy in `./rewards.txt`.

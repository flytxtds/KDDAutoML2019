# KDDAutoML2019


This repository contains solution for AutoML in binary classification problems for temporal relational data. It was developed by a joint team ('autoflylearn') from Flytxt as a part of KDDCUP 2019 AutoML Challenge (The 5th AutoML Challenge:
AutoML for Temporal Relational Data). Our solution improved over the baseline solution provided by the organizers significantly and one of the prominent solutions.

Team:

1.Harshvardhan Solanki (harshvardhan.solanki@flytxt.com)
2.Binay Gupta (binay.gupta@flytxt.com)
3.Amit Kumar Meher (amit.meher@flytxt.com)
4.Nasibullah Ohidullah(nasibullah104@gmail.com)





Contents:
---------

ingestion_program/: The code and libraries used on Codalab to run your submmission.
scoring_program/: The code and libraries used on Codalab to score your submmission.
sample_code_submission/: Our solution code
sample_data/: Some sample data to test code.
sample_ref/: Reference data required to evaluate your submission.





How to run:
-----------

1. Install docker from https://docs.docker.com/get-started/.
2. At the shell, change to the startingkit directory, run

```
docker run -it --rm -u root -v $(pwd):/app/kddcup codalab/codalab-legacy:py3 bash
```

3. Now your are in the bash of the docker container, run ingestion program

```
cd /app/kddcup
python3 ingestion_program/ingestion.py
```
It runs sample_code_submission and the predictions will be in sample_predictions directory

4. Now run scoring program:

```
python3 scoring_program/score.py
```

It will score the predictions and the results will be in sample_scoring_output directory

### Remark


- The full call of the ingestion program is:

```
python3 ingestion_program/ingestion.py local sample_data sample_predictions ingestion_program sample_code_submission
```

- The full call of the scoring program is:

```
python3 scoring_program/score.py local sample_predictions sample_ref sample_scoring_output
```

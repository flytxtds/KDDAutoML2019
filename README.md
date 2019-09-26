# KDDAutoML2019


This repository contains solution for AutoML in binary classification problems for temporal relational data. It was developed by a joint team ('autoflylearn') from Flytxt as a part of KDDCUP 2019 AutoML Challenge (The 5th AutoML Challenge:
AutoML for Temporal Relational Data). Our solution improved significantly over the baseline solution provided by the organizers significantly and was one of the prominent solutions.

Team:

1. Harshvardhan Solanki (harshvardhan.solanki@flytxt.com)
2. Binay Gupta (binay.gupta@flytxt.com)
3. Amit Kumar Meher (amit.meher@flytxt.com)
4. Nasibullah Ohidullah(nasibullah104@gmail.com)





Contents:
---------


- sample_code_submission/: Our solution code





How to run:
-----------

1. Install docker from https://docs.docker.com/get-started/.
2. Download starter-kit from competition organizer's website and replace their sample code solution with our folder "sample_code_submission"

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

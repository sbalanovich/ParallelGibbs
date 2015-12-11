# ParallelGibbs
CS205 Final Project

Andrew Mauboussin and Serguei Balanovich


How to run
---


First, install the dependencies listed in requirements.txt. You can set up a virtual environment for the project with all of the required dependencies by opening a terminal shell inside the main directory and running the following commands (this requires having virtualenv installed; you can `pip install virtualenv` if you don't already have it).
```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

Once you have the dependencies, you run the project using a command line interface by running `python driver.py`.

```
Options:
  --dataset TEXT        Dataset to run sampler on. Options include synthetic,
                        reuters, nytimes.
  --method TEXT         Sampler implementation to be run. Options include gpu,
                        serial, multiprocessing.
  --n_topics INTEGER    Number of topics (k in the literature)
  --P INTEGER           Number of processes (ignored for serial version)
  --iterations INTEGER  Number of iterations to run sampler for
  --help                Show this message and exit.
```
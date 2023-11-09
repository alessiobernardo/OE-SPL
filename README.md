# MOA (Massive Online Analysis)
[![Build Status](https://travis-ci.org/Waikato/moa.svg?branch=master)](https://travis-ci.org/Waikato/moa)
[![Maven Central](https://img.shields.io/maven-central/v/nz.ac.waikato.cms.moa/moa-pom.svg)](https://mvnrepository.com/artifact/nz.ac.waikato.cms)
[![DockerHub](https://img.shields.io/badge/docker-available-blue.svg?logo=docker)](https://hub.docker.com/r/waikato/moa)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

![MOA][logo]

[logo]: http://moa.cms.waikato.ac.nz/wp-content/uploads/2014/11/LogoMOA.jpg "Logo MOA"

MOA is the most popular open source framework for data stream mining, with a very active growing community ([blog](http://moa.cms.waikato.ac.nz/blog/)). It includes a collection of machine learning algorithms (classification, regression, clustering, outlier detection, concept drift detection and recommender systems) and tools for evaluation. Related to the WEKA project, MOA is also written in Java, while scaling to more demanding problems.

http://moa.cms.waikato.ac.nz/

## Using MOA

* [Getting Started](http://moa.cms.waikato.ac.nz/getting-started/)
* [Documentation](http://moa.cms.waikato.ac.nz/documentation/)
* [About MOA](http://moa.cms.waikato.ac.nz/details/)

MOA performs BIG DATA stream mining in real time, and large scale machine learning. MOA can be extended with new mining algorithms, and new stream generators or evaluation measures. The goal is to provide a benchmark suite for the stream mining community. 

## OE-SPL
The OE-SPL algorithm is in the `moa/src/main/java/moa/classifiers/meta` folder.

We run all the experiments on a virtual machine inside a Docker container to correctly extract the memory consumed by each model. After launching the Docker container, to reproduce all the experiments of the papar, run the two `.sh` files, where each line corresponds to a single model tested on a particular data stream. For example, the following line shows the `first` run of the `OE-SPL` model with `HT` as base learner using the `sea` stream having concept drift type `P(y)`. The output is then redirected to the `results` folder. Each model, for each data stream and base learner, was tested `10` times with a different seed value.

`
docker run --rm --name="P_y_sea_HT_1" -v $(pwd)/results:/src/results test_moa bash -c "java -Xmx15g -Xss50M -cp moa.jar -javaagent:sizeofag-1.0.4.jar moa.DoTask \"EvaluatePrequential -l (trees.HoeffdingTree -S -D 32) -s (ArffFileStream -f (datasets/P(y)/sea.arff)) -e (WindowFixedClassificationPerformanceEvaluator -w 50000 -o) -i -1 -f 1000\" 1> results/P\(y\)/sea/HT/1.csv 2> results/P\(y\)/sea/HT/1_err.csv"
`

Finally, we averaged the results over the `10` repetitions of all the models tested, and we used those results to apply the Welch's t-test and the Nemeyi test.

## Citing OE-SPL
If you want to refer to OE-SPL in a publication, please cite the following paper:

> Alessio Bernardo, Emanuele Della Valle, and Albert Bifet. "Choosing the Right Time to Learn Evolving Data Streams." 2023 IEEE International Conference on Big Data (Big Data). IEEE, 2023 (To Appear).
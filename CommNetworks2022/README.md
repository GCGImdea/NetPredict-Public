# Explainable Machine Learning for Anomaly Detection and Classification in Moblile Networks

[Juan Marcos Ramírez](https://juanmarcosramirez.github.io/), Pablo Rojo, Fernando Díez, [Vincenzo Mancuso](https://networks.imdea.org/team/imdea-networks-team/people/vincenzo-mancuso/), and [Antonio Fernández-Anta](https://networks.imdea.org/team/imdea-networks-team/people/antonio-fernandez-anta/)

## Abstract

Mobile communication providers continuously collect many parameters, statistics, and key performance indicators (KPIs) with the goal of identifying operation scenarios that can affect the quality of Internet-based services. In this regard, anomaly detection and classification in mobile networks have become challenging tasks due to both the huge number of involved variables and the unknown distributions exhibited by input features. This paper introduces an unsupervised methodology based on both a data cleaning strategy and explainable machine learning models to detect and classify performance anomalies in mobile networks. Specifically, this methodology dubbed explainable machine learning for anomaly detection and classification (XMLAD) aims at identifying features and operation scenarios characterizing performance anomalies without resorting to parameter tuning. To this end, this approach includes a data cleaning stage that extracts and removes outliers from experiments and features to train the anomaly detection engine with the cleanest possible dataset. Moreover, the methodology considers the differences between discretized values of the target KPI and labels predicted by the anomaly detection engine to build the anomaly classification engine which identifies features and thresholds that could cause performance anomalies. The proposed methodology incorporates two decision tree classifiers to build explainable models of the anomaly detection and classification engines whose decision structures recognize features and thresholds describing both normal behaviors and performance anomalies. We evaluate the XMLAD methodology on real datasets captured by operational tests in commercial networks. In addition, we present a testbed that generates synthetic data using a known TCP throughput model to assess the accuracy of the proposed approach.

![Flowchart](https://github.com/GCGImdea/NetPredict/blob/main/CommNetworks2022/images/XMLAD_Figure.png?raw=true "Flowchart")


|     |     |     |
| :-: | :-: | :-: |
|     |     |     |
| Clustering 2D | Discretization | Model 1 optimization |
| ![](https://github.com/GCGImdea/NetPredict/blob/main/CommNetworks2022/images/nokia_dataset/FL_clustering2D.png?raw=true "Flowchart") | ![](https://github.com/GCGImdea/NetPredict/blob/main/CommNetworks2022/images/nokia_dataset/FL_discretization.png?raw=true "Flowchart") | ![](https://github.com/GCGImdea/NetPredict/blob/main/CommNetworks2022/images/nokia_dataset/FL_model1_optimization.png?raw=true "Flowchart") |
| Differences | Miscoding | Model 2 optimization |
| ![](https://github.com/GCGImdea/NetPredict/blob/main/CommNetworks2022/images/nokia_dataset/FL_differences.png?raw=true "Flowchart") | ![](https://github.com/GCGImdea/NetPredict/blob/main/CommNetworks2022/images/nokia_dataset/FL_miscoding.png?raw=true "Flowchart") | ![](https://github.com/GCGImdea/NetPredict/blob/main/CommNetworks2022/images/nokia_dataset/FL_model2_optimization.png?raw=true "Flowchart") |

![Decision tree](https://github.com/GCGImdea/NetPredict/blob/main/CommNetworks2022/models/nokia_dataset/model1_file_dl.png?raw=true "Decision tree")

![Decision tree](https://github.com/GCGImdea/NetPredict/blob/main/CommNetworks2022/models/nokia_dataset/model2_file_dl.png?raw=true "Decision tree")

### Platform

* Ubuntu 20.04 OS.

### License

This code package is licensed under the GNU GENERAL PUBLIC LICENSE (version 3) - see the [LICENSE](LICENSE) file for details. 

### Contact

[Juan Marcos Ramirez](juan.ramirez@imdea.org). Postdoctoral Researcher. [IMDEA Networks Institute](https://networks.imdea.org/). Madrid, 28918, Spain. 

### Date

October 03rd, 2022

### Acknowledgements

This work has been partially supported by the Project AEON-CPS (TSI-063000-2021-38), funded by the Ministry of Economic Affairs and Digital Transformation and the European Union NextGeneration-EU in the framework of the Spanish Recovery, Transformation and Resilience Plan. This work has been also supported by the Project EDGEDATA-CM (S2018/TCS-4499), funded by the Department of Education and Research of the Regional Government of Madrid, through the 2018 R&D technology program for research groups, co-financed by the Operational Programs of the European Social Fund (ESF) and the European Regional Development Fund (ERDF). The work of Juan M. Ramirez has been supported by the Project ECID: Edge Computing for Intelligent Driving (PID2019-109805RB-I00) funded by the Spanish State Research Agency, Spanish Ministry of Science and Innovation. This paper has been supported by the projects Netpredict4 and Netpredict5 funded by Nokia Spain .

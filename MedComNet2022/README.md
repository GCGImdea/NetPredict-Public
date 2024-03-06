# Cleaning Matters!  Preprocessing-enhanced Anomaly Detection and Classification in Mobile Networks

[Juan Marcos Ramírez](https://juanmarcosramirez.github.io/), Pablo Rojo, Fernando Díez, [Vincenzo Mancuso](https://networks.imdea.org/team/imdea-networks-team/people/vincenzo-mancuso/), and [Antonio Fernández-Anta](https://networks.imdea.org/team/imdea-networks-team/people/antonio-fernandez-anta/)

## Abstract

Mobile communications providers often monitor key performance indicators (KPIs) with the goal of identifying anomalous operation scenarios that can affect the quality of Internet-based services. In this regard, anomaly detection and classification in mobile networks has become a challenging task due to the unknown distributions exhibited by the collected data and the lack of interpretability of the embedded (machine learning) models. This paper proposes an unsupervised end-to-end methodology based on both a data cleaning strategy and explainable machine learning models to detect and classify performance anomalies in mobile networks. The proposed approach, dubbed clean and explainable anomaly detection and classification (KLNX), aims at identifying attributes and operation scenarios that could induce anomalous KPI values without resorting to parameter tuning. Unlike previous methodologies, the proposed method includes a data cleaning stage that extracts and removes experiments and attributes considered outliers in order to train the anomaly detection engine with the cleanest possible dataset. Additionally, machine learning models provide interpretable information about features and boundaries describing both the normal network behavior and the anomalous scenarios. To evaluate the performance of the proposed method, a testbed generating synthetic data is developed using a known TCP throughput model. Finally, the methodology is assessed on a real data set captured by operational tests in commercial networks.

![Flowchart](https://github.com/GCGImdea/NetPredict/blob/main/MedComNet2022/images/KLNX_Flowchart.png?raw=true "Flowchart")

![Decision tree](https://github.com/GCGImdea/NetPredict/blob/main/MedComNet2022/models/model2_realdata.png?raw=true "Decision tree")

### Platform

* Ubuntu 20.04 OS.

### License

This code package is licensed under the GNU GENERAL PUBLIC LICENSE (version 3) - see the [LICENSE](LICENSE) file for details. 

### Contact

[Juan Marcos Ramirez](juan.ramirez@imdea.org). Postdoctoral Researcher. [IMDEA Networks Institute](https://networks.imdea.org/). Madrid, 28918, Spain. 

### Date

May 12, 2022

### Acknowledgements

The work was supported by the DiscoLedger project (PDC2021-121836-I00) funded by MCIN/AEI /10.13039/501100011033 and the European Union through the Next Generation EU/ PRTR program.

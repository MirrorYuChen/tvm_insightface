# **tvm_insightface**
## **face recognize based on tvm**
## **the source model comes from insightface:**
## https://github.com/deepinsight/insightface.git
# **How to run?**
## **1. configure your tvm environmental**
## **you can refer to my csdn blog:**
## https://blog.csdn.net/sinat_31425585/article/details/89395680
## **2. >> python compile.py **  
## then three files will be produced: deploy_graph.json, deploy_lib.so, deploy_param.params
## **3. >> python deploy.py**
## you 'll get the similarity between two face pictures

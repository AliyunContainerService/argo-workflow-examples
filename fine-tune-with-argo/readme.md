![img](https://github.com/AliyunContainerService/argo-workflow-examples/blob/main/fine-tune-with-argo/fine-tune.png)
1. Understand [Alibaba Cloud Argo Workflow Cluster](https://help.aliyun.com/zh/ack/distributed-cloud-container-platform-for-kubernetes/user-guide/overview-12)
2. Create [Alibaba Cloud Argo Workflow Cluster](https://help.aliyun.com/zh/ack/distributed-cloud-container-platform-for-kubernetes/user-guide/create-a-workflow-cluster), and get cluster kubeconfig from console.
3. Setup [PV/PVC for Alibaba OSS FS](https://help.aliyun.com/zh/ack/distributed-cloud-container-platform-for-kubernetes/user-guide/use-volumes), config ak, sk, bucket, bucket url in oss-pvpvc.yaml, and run kubectl apply -f oss-pvpvc.yaml
4. Submit Workflow
    option 1: pip install hera, python tcm-fine-tune.py 
    option 2: argo submit tcm-fine-argo-workflow.yaml

       

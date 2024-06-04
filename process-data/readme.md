![img](https://github.com/AliyunContainerService/argo-workflow-examples/blob/main/process-data/demo.png)
1. Understand [Alibaba Cloud Argo Workflow Cluster](https://help.aliyun.com/zh/ack/distributed-cloud-container-platform-for-kubernetes/user-guide/overview-12)
2. Create [Alibaba Cloud Argo Workflow Cluster](https://help.aliyun.com/zh/ack/distributed-cloud-container-platform-for-kubernetes/user-guide/create-a-workflow-cluster), and get cluster kubeconfig from console.
3. Setup [PV/PVC for Alibaba OSS FS](https://help.aliyun.com/zh/ack/distributed-cloud-container-platform-for-kubernetes/user-guide/use-volumes), config ak, sk, bucket, bucket url in oss-pvpvc.yaml, and run kubectl apply -f oss-pvpvc.yaml
4. Create "aggregation-demo" Dir in OSS to store demo files.
5. Run command : argo submit aggregation-createfile.yaml to prepare data, which creating 512 files in oss bucket.
6. Run command : kubectl apply -f aggregation-template.yaml to create workflow template.
7. Run command : argo submit --from wftmpl/aggregation-demo, to create workflow from workflow template to process files in oss.
8. You can go to argo workflow console to check status.
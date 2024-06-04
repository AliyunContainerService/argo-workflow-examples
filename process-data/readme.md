1. Understand [Alibaba Cloud Argo Workflow Cluster](https://help.aliyun.com/zh/ack/distributed-cloud-container-platform-for-kubernetes/user-guide/overview-12)
2. Create [Alibaba Cloud Argo Workflow Cluster](https://help.aliyun.com/zh/ack/distributed-cloud-container-platform-for-kubernetes/user-guide/create-a-workflow-cluster)
3. Setup [Artifact with Alibaba Cloud OSS](https://help.aliyun.com/zh/ack/distributed-cloud-container-platform-for-kubernetes/user-guide/configure-artifacts)
4. Create "aggregation-demo" Dir in OSS
5. Run command : argo submit aggregation-createfile.yaml to prepare data.
6. Run command : kubectl apply -f aggregation-template.yaml to create workflow template.
7. Run command : argo submit --from wftmpl/aggregation-demo
User case:
If you have large volume data in nas which need to process by mulitple pods in parallel, it will be easy to hit the limitation of nas throughput. You can refer this example. 
1. copy data from nas to cloud disk.
2. create snapshot from cloud disk.
3. create multiple workflow pods to read data from snapshot. For each pod, a pvc will be created from the snapshot and moute to the pod. PVC will be removed when pod is removed.


Steps:
1. create nas pv: kubectl apply -f nas/pv-nas.yaml 
2. create nas pvc: kubectl apply -f nas/pvc-nas.yaml
3. create disk pvc: kubectl apply -f disk/pvc-disk.yaml
5. create workflow to create cloud disk and copy data from nas to disk: 

6. create snapshot storage class: kubectl apply -f snapshot/volume-snapshot-class.yaml
7. create snapshot: kubectl apply -f snapshot/volume-snapshot.yaml
8. create workflow to read data from ephemeral cloud disk, the disk is created from the snapshot: argo submit parallel-read-snapshot-data.yaml

Debug commands:
1. get snapshot storage class: kubectl get volumesnapshotclasses
2. get snapshot: kubectl get volumesnapshots, and check from alibabacloud ecs console.
3. delete snapshot: kubectl delete volumesnapshots xxx, and check from alibabacloud ecs console.
4. delete workflow parallel-read-snapshot-data, cloud disk will be removed
5. delete disk pvc pvc-disk, cloud disk will be removed

Reference doc link in alibaba cloud:
https://help.aliyun.com/zh/ack/distributed-cloud-container-platform-for-kubernetes/user-guide/overview-12
https://help.aliyun.com/zh/ack/ack-managed-and-ack-dedicated/user-guide/use-volume-snapshots-created-from-disks
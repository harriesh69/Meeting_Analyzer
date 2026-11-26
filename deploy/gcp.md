# Deploying to Google Cloud

This runs the whole stack on Kubernetes (k3s) on a single Compute Engine VM.
k3s is a full, certified Kubernetes distribution that fits on one machine,
so the same manifests also work on GKE or any other cluster.

New Google Cloud accounts get free trial credit, which covers running this
VM for a long time. Stop the VM when you are not using it.

## 1. Create the VM

```bash
gcloud config set project YOUR_PROJECT_ID

gcloud compute instances create meeting-analyzer \
  --zone=asia-south1-a \
  --machine-type=e2-standard-4 \
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
  --boot-disk-size=60GB \
  --tags=http-server

gcloud compute firewall-rules create allow-http \
  --allow=tcp:80 --target-tags=http-server
```

`e2-standard-4` has 4 vCPUs and 16 GB of memory: enough for Whisper `small`,
pyannote and t5 on CPU. CPU inference is slow on long recordings, so time a few of
your own and note the numbers; `ma_job_duration_seconds` in `/metrics` records them.

## 2. Install Docker and k3s

```bash
gcloud compute ssh meeting-analyzer --zone=asia-south1-a

curl -fsSL https://get.docker.com | sudo sh
curl -sfL https://get.k3s.io | sh -
sudo chmod 644 /etc/rancher/k3s/k3s.yaml
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
kubectl get nodes        # should show one Ready node
```

## 3. Build the images and load them into k3s

```bash
git clone https://github.com/harriesh69/Meeting_Analyzer && cd Meeting_Analyzer
for i in api worker ui; do
  sudo docker build -f docker/$i.Dockerfile -t meeting-analyzer-$i:v1 .
  sudo docker save meeting-analyzer-$i:v1 | sudo k3s ctr images import -
done
```

## 4. Add the Hugging Face token and deploy

The pyannote model is gated: accept its terms on Hugging Face and create a
read token first (see the main README).

```bash
kubectl create namespace meeting-analyzer
kubectl -n meeting-analyzer create secret generic hf-token --from-literal=token=hf_YOUR_TOKEN
kubectl apply -k k8s/overlays/cloud
kubectl -n meeting-analyzer get pods -w     # the worker takes a few minutes to load models
```

## 5. Use it

```bash
EXTERNAL_IP=$(gcloud compute instances describe meeting-analyzer --zone=asia-south1-a \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
echo "UI:  http://$EXTERNAL_IP/"
echo "API: http://$EXTERNAL_IP/docs"

curl -F file=@meeting.m4a http://$EXTERNAL_IP/v1/meetings
curl http://$EXTERNAL_IP/v1/meetings/JOB_ID
```

## Stop and clean up

```bash
gcloud compute instances stop meeting-analyzer --zone=asia-south1-a     # pause billing for compute
gcloud compute instances delete meeting-analyzer --zone=asia-south1-a   # remove everything
```

## Moving to GKE

Push the images to Artifact Registry, point the overlay at them with
`kustomize edit set image`, raise the worker HPA's `maxReplicas`, and apply
the same overlay. For scale-to-zero workers, install KEDA and use
`k8s/optional/keda-scaledobject.yaml`.

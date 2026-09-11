# Chest X-ray Pneumonia Classifier — Deployment (A2)

Deploys the fine-tuned ResNet18 from A1 as a public web service: a JSON
prediction API, a Docker image, and a web page for submitting X-rays.

**Live URL:** _paste your Cloud Run URL here once deployed_

> Screening aid for coursework. Not a diagnostic device, and not usable on real
> patients.

---

## What this serves

| | |
|---|---|
| Model | ResNet18, ImageNet-pretrained, fine-tuned end-to-end on 5,856 chest X-rays |
| Head | `nn.Linear(512, 1)` — one logit, sigmoid, binary |
| Input | Grayscale → 3 channels, resized to 224×224, ImageNet normalisation |
| Threshold | 0.50 |
| Test performance | 84.9% accuracy · 0.822 macro F1 · 0.973 ROC-AUC |
| Known weakness | NORMAL recall 60.7% — it over-flags healthy chests as pneumonia |

The preprocessing in `app/model.py` is a line-for-line copy of `eval_transform`
from A1 Step 3. If you change one, change the other, or the served predictions
stop matching the model you evaluated.

A1's Step 5 tuned the decision threshold and found the val-optimal value (0.52)
flipped zero test predictions, so 0.50 ships. It's exposed as the
`DECISION_THRESHOLD` environment variable if you want to move it without a
rebuild.

---

## Layout

```
.
├── app/
│   ├── main.py            FastAPI app, routes, request logging
│   ├── model.py           checkpoint loading + inference
│   ├── images.py          URL fetching and decoding, with limits
│   ├── config.py          environment variables
│   ├── logging_config.py  JSON logs for Cloud Logging
│   ├── schemas.py         request/response models
│   └── static/index.html  the web UI
├── models/
│   └── resnet18_finetuned.pt   ← copy this in from A1
├── Dockerfile
├── .dockerignore
├── requirements.txt            serving deps (6 packages)
├── requirements-torch.txt      torch, from the CPU wheel index
└── test_api.py                 end-to-end test script
```

---

## Step 0 — Add the model

The repo is inert without it:

```bash
cp /path/to/A1/models/resnet18_finetuned.pt models/
```

The Dockerfile's `COPY` fails the build if it's missing. That's deliberate —
better a build error than a container that starts fine and 503s on every
request.

---

## Step 1 — The API

Four endpoints, plus generated docs at `/docs`.

**`POST /predict`** — classify by URL

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/chest-xray.jpeg"}'
```

```json
{
  "prediction": "PNEUMONIA",
  "label_index": 1,
  "probability_pneumonia": 0.973214,
  "confidence": 0.973214,
  "class_probabilities": { "NORMAL": 0.026786, "PNEUMONIA": 0.973214 },
  "threshold": 0.5,
  "model_version": "resnet18-finetuned-a1",
  "inference_ms": 84.21
}
```

**`POST /predict/file`** — classify an upload; this is what the web page calls

```bash
curl -X POST http://localhost:8080/predict/file -F "file=@chest-xray.jpeg"
```

**`GET /health`** — liveness. Always 200 while the process is up;
`model_loaded` tells you whether it can actually predict. Startup logs the
failure and keeps serving rather than crash-looping, so a broken checkpoint
shows up in `/health` instead of a restart loop.

**`GET /metadata`** — version, threshold, classes, size limits. The UI reads
this so the threshold marker on the gauge follows whatever the server is
actually using.

### Errors

| Status | When |
|---|---|
| 400 | Not an image, too large, bad URL scheme, unreachable host, private address |
| 422 | `image_url` missing from the body (FastAPI validation) |
| 500 | Inference threw |
| 503 | Model never loaded |

### Run it locally without Docker

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cpu -r requirements-torch.txt
pip install -r requirements.txt
LOG_FORMAT=text uvicorn app.main:app --reload --port 8080
```

Then open <http://localhost:8080>. `LOG_FORMAT=text` gives readable logs; leave
it off in the container, where JSON is what you want.

## Test the app.
Test with possible URL : < https://upload.wikimedia.org/wikipedia/commons/8/83/Chest_X-ray_2346.jpg?utm_source=commons.wikimedia.org&utm_campaign=index&utm_content=original> 
Run test with any image (.jpeg  / .png ) from your local collection of chest x-rays.

Or any other Jpeg image from files or web.
---

## Step 2 — Docker

```bash
docker build -t xray-api .
docker run -p 8080:8080 xray-api
or
docker run --rm -p 8080:8080 xray-api
```

Then <http://localhost:8080>, and:

```bash
python test_api.py http://localhost:8080
```

Three things in the Dockerfile worth understanding, because they're the usual
places this goes wrong:

**CPU-only torch.** The default PyPI `torch` wheel bundles CUDA — about 2 GB of
GPU libraries a Cloud Run instance has no card to use. Installing from
`--index-url https://download.pytorch.org/whl/cpu` cuts that to roughly 200 MB.

**`weights=None` in `model.py`.** A1 built the model with
`models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`, which downloads
ImageNet weights on first call. In a container that means a network fetch at
every cold start, and a crash if egress is blocked. Here the architecture is
built empty and every parameter comes from your checkpoint.

**Dependency layers before code.** `requirements*.txt` is copied and installed
before `app/`, so editing `main.py` rebuilds in seconds instead of
re-downloading PyTorch.

Building on an Apple Silicon Mac? Cloud Run runs amd64, and an arm64 image will
deploy and then fail to start with an exec format error:

```bash
docker build --platform linux/amd64 -t xray-api .
```

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | 8080 | Listening port. Cloud Run injects this; don't hard-code it. |
| `MODEL_PATH` | `/app/models/resnet18_finetuned.pt` | Checkpoint location |
| `DECISION_THRESHOLD` | 0.5 | Probability cutoff for PNEUMONIA |
| `MODEL_VERSION` | `resnet18-finetuned-a1` | Reported in every response |
| `TORCH_NUM_THREADS` | 1 | Keep at 1 on a 1-vCPU instance; more threads means contention, not speed |
| `LOG_LEVEL` | INFO | |
| `LOG_FORMAT` | json | `text` for local development |
| `MAX_IMAGE_BYTES` | 10485760 | 10 MB ceiling on fetches and uploads |
| `ALLOWED_ORIGINS` | `*` | CORS. Narrow this if the page moves off this origin. |
| `BLOCK_PRIVATE_ADDRESSES` | true | See the note below |

Override at runtime, no rebuild:

```bash
docker run -p 8080:8080 -e DECISION_THRESHOLD=0.35 -e LOG_FORMAT=text xray-api
```

### One security note

`/predict` fetches URLs on the caller's behalf. Once it's publicly reachable,
anyone can ask it to fetch anything the container can reach — including cloud
metadata endpoints, which on GCP and AWS hand out credentials to whoever asks
from inside the instance. `app/images.py` resolves the hostname and refuses
private, loopback, and link-local addresses before making the request, and caps
response size and time. Leave `BLOCK_PRIVATE_ADDRESSES=true` on anything with a
public URL.

---

## Step 3 — Deploy to Cloud Run

Replace `YOUR_PROJECT_ID` throughout. Any region works;
`northamerica-northeast1` is Montréal.

```bash
export PROJECT_ID=YOUR_PROJECT_ID
export REGION=northamerica-northeast1
export SERVICE=xray-api

gcloud auth login
gcloud config set project $PROJECT_ID

gcloud services enable run.googleapis.com \
                       artifactregistry.googleapis.com \
                       cloudbuild.googleapis.com
```

**Create the Artifact Registry repo:**

```bash
gcloud artifacts repositories create ml-models \
  --repository-format=docker \
  --location=$REGION \
  --description="A2 deployment images"
```

**Build and push.** Easiest path — Cloud Build does it server-side, so you don't
need a local Docker daemon or a fast uplink, and architecture stops mattering:

```bash
gcloud builds submit \
  --tag $REGION-docker.pkg.dev/$PROJECT_ID/ml-models/$SERVICE:v1
```

<details>
<summary>Or push a locally built image instead</summary>

```bash
gcloud auth configure-docker $REGION-docker.pkg.dev

docker build --platform linux/amd64 \
  -t $REGION-docker.pkg.dev/$PROJECT_ID/ml-models/$SERVICE:v1 .

docker push $REGION-docker.pkg.dev/$PROJECT_ID/ml-models/$SERVICE:v1
```
</details>

**Deploy:**

```bash
gcloud run deploy $SERVICE \
  --image $REGION-docker.pkg.dev/$PROJECT_ID/ml-models/$SERVICE:v1 \
  --region $REGION \
  --platform managed \
  --allow-unauthenticated \
  --port 8080 \
  --memory 2Gi \
  --cpu 1 \
  --timeout 120 \
  --concurrency 4 \
  --min-instances 0 \
  --max-instances 3 \
  --set-env-vars "DECISION_THRESHOLD=0.5,TORCH_NUM_THREADS=1,LOG_LEVEL=INFO"
```

`--allow-unauthenticated` is what makes the endpoint public — without it every
request needs an IAM token and your grader gets a 403.

The command prints a `https://xray-api-….run.app` URL. Put it at the top of this
README, then:

```bash
python test_api.py https://xray-api-xxxxx.run.app
```

**Why those numbers.** 2 GiB because torch plus ResNet18 weights sits around
1 GB resident and the default 512 MiB kills the container during load.
Concurrency 4 because inference is CPU-bound with one torch thread — piling more
concurrent requests onto one instance just queues them. Max instances 3 as a
cost ceiling on a student project.

Cold starts run 10–20 seconds: the container has to import torch and load 45 MB
of weights. `--min-instances 1` removes that but bills you for an always-warm
instance. For a graded demo, hit the URL once a minute before showing it.

### Redeploying

```bash
gcloud builds submit --tag $REGION-docker.pkg.dev/$PROJECT_ID/ml-models/$SERVICE:v2
gcloud run deploy $SERVICE --image $REGION-docker.pkg.dev/$PROJECT_ID/ml-models/$SERVICE:v2 --region $REGION
```

Increment the tag rather than reusing `:v1` or `:latest` — Cloud Run keeps each
revision, so a numbered tag lets you roll back to a known-good one.

---

## Step 4 — Web UI

Served at `/` by the same container, so there's no second thing to deploy and no
CORS to configure. Drop an X-ray on the viewer or paste a URL, and the result
comes back with the probability drawn against the decision threshold — the
vertical mark on the gauge is the 0.50 cutoff, so you can see how close a call
was rather than just which side it landed on. Anything within 0.10 of the
threshold gets flagged as borderline.

The page reads `/metadata` on load, so the threshold marker tracks whatever the
server is configured with.

---

## Step 5 — Monitoring

**Logs.** Every request produces one JSON line with method, path, status,
duration, and request ID; predictions add the class, probability, and inference
time. Cloud Logging parses the `severity` field, so errors show up red and you
can filter on any field.

```bash
# stream
gcloud run services logs tail $SERVICE --region $REGION

# recent
gcloud run services logs read $SERVICE --region $REGION --limit 50

# just the predictions
gcloud logging read \
  'resource.type="cloud_run_revision" AND jsonPayload.message="Prediction served"' \
  --limit 20 --format json
```

Console: **Cloud Run → your service → Logs**.

**Dashboard.** **Cloud Run → your service → Metrics** — request count, latency
percentiles, instance count, memory and CPU utilisation.

**Screenshots to capture for the deliverable:**

1. Cloud Run service page, showing the URL and a green revision
2. Logs tab with prediction entries visible
3. Metrics tab after you've sent some traffic
4. `python test_api.py https://your-url` output, all checks passing
5. The web UI with a real prediction on screen
6. Artifact Registry showing the pushed image

Send traffic before screenshotting the metrics, or the charts are empty:

```bash
for i in {1..15}; do python test_api.py https://your-url > /dev/null; done
```

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| Build fails at `COPY models/...` | Checkpoint not copied in — see Step 0 |
| `exec format error` on deploy | arm64 image on amd64 Cloud Run. Rebuild with `--platform linux/amd64`, or use `gcloud builds submit` |
| Container fails to start, no useful log | Almost always memory. Raise to 2Gi |
| `libgomp.so.1: cannot open shared object file` | `libgomp1` missing — the Dockerfile installs it; check you didn't drop the apt line |
| `Error loading state_dict` | torch/torchvision version drift between training and serving. Pin `requirements-torch.txt` to your A1 versions |
| 403 on the public URL | Missing `--allow-unauthenticated` |
| First request takes 20s, rest are fast | Normal cold start |
| `/health` says `model_loaded: false` | Check startup logs; the loader logs the reason and keeps serving so you can read it |

---

## What was deliberately left out

The A1 `requirements.txt` has ~180 packages: jupyter, matplotlib, seaborn,
spacy, gensim, scikit-learn, kaggle, lime, wordcloud. None of it runs at
inference time. Serving needs a web server, an image decoder, and the model
runtime, which is six packages plus torch. Everything omitted is image size not
downloaded on every cold start, and a dependency not inherited.

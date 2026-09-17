# Chest X-ray Pneumonia Classifier — Deployment (A2)

Serves the fine-tuned ResNet18 from A1 as a public web service: a JSON prediction
API, a containerised build, and a web page for submitting X-rays.

**Live service:** https://xray-api-747277387717.northamerica-northeast1.run.app
**API docs:** https://xray-api-747277387717.northamerica-northeast1.run.app/docs
**Repository:** https://github.com/DanGherghiceanu/A2-deployment

> Screening aid built for coursework. Not a diagnostic device, and not for use on
> real patients.

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

That last row is the honest headline. The model is strong at finding pneumonia
and weak at confirming a chest is clear, so a PNEUMONIA result is a prompt to
look closer and a NORMAL result is the weaker of its two calls. The web UI says
so on screen rather than only in this file.

The preprocessing in `app/model.py` is a line-for-line copy of `eval_transform`
from A1 Step 3. If one changes, the other must change with it, or the served
predictions stop matching the model that was evaluated.

A1's Step 5 tuned the decision threshold and found the val-optimal value (0.52)
flipped zero test predictions, so 0.50 ships. It's exposed as the
`DECISION_THRESHOLD` environment variable, so the operating point can move at
deploy time without a rebuild.

---

## Layout

```
.
├── app/
│   ├── __init__.py        makes app/ a package — the imports fail without it
│   ├── main.py            FastAPI app, routes, request logging
│   ├── model.py           checkpoint loading + inference
│   ├── images.py          URL fetching and decoding, with limits
│   ├── config.py          environment variables
│   ├── logging_config.py  JSON logs for Cloud Logging
│   ├── schemas.py         request/response models
│   └── static/
│       └── index.html     the web UI
├── models/
│   └── resnet18_finetuned.pt   44.8 MB, copied in from A1
├── samples/
│   └── xray-pneumonia.jpeg     public test image for the API
├── Dockerfile
├── .dockerignore
├── requirements.txt            serving deps — 6 packages
├── requirements-torch.txt      torch, from the CPU wheel index
└── test_api.py                 end-to-end test script
```

This is a separate project root from A1, not a subfolder of it. The reason is
the build context: `docker build .` tars up the entire directory before building,
so rooting it here means ~45 MB, while rooting it in the A1 folder would mean
uploading the chest X-ray dataset and a `.venv` full of jupyter and spacy on
every build.

---

## Step 0 — Add the model

The repo is inert without the checkpoint:

```powershell
copy "resnet18_finetuned.pt" into models\
```

The Dockerfile's `COPY` fails the build if it's missing. That's deliberate —
better a build error than a container that starts cleanly and 503s on every
request.

The file is committed to git at 44.8 MB. GitHub warns above 50 MB and hard-fails
at 100 MB, so this fits, but a production setup would keep weights in object
storage or Git LFS and version them separately from the code.

> GitHub displays this as **42.7 MB** — it counts in binary mebibytes (÷ 1024²)
> while the container logs report 44.8 in decimal megabytes (÷ 10⁶). Same file.
> If it ever shows as ~130 bytes instead, Git LFS replaced it with a pointer and
> the real weights aren't in the repo.

---

## Step 1 — The API

Four endpoints, plus generated docs at `/docs`.

**`POST /predict`** — classify by URL

```bash
curl -X POST https://xray-api-747277387717.northamerica-northeast1.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://raw.githubusercontent.com/DanGherghiceanu/A2-deployment/main/samples/xray-pneumonia.jpeg"}'
```

```json
{
  "prediction": "PNEUMONIA",
  "label_index": 1,
  "probability_pneumonia": 0.986100,
  "confidence": 0.986100,
  "class_probabilities": { "NORMAL": 0.013900, "PNEUMONIA": 0.986100 },
  "threshold": 0.5,
  "model_version": "resnet18-finetuned-a1",
  "inference_ms": 84.21
}
```

**`POST /predict/file`** — classify an upload; this is what the web page calls

```bash
curl -X POST https://xray-api-747277387717.northamerica-northeast1.run.app/predict/file \
  -F "file=@chest-xray.jpeg"
```

**`GET /health`** — liveness. Returns 200 while the process is up; `model_loaded`
tells you whether it can actually predict. Startup logs a load failure and keeps
serving rather than crash-looping, so a broken checkpoint surfaces in `/health`
instead of a restart loop with the reason buried in the logs.

**`GET /metadata`** — version, threshold, classes, size limits. The UI reads this
on load, so the threshold marker on the gauge follows whatever the server is
actually configured with.

### Errors

| Status | When |
|---|---|
| 400 | Not an image, too large, bad URL scheme, unreachable host, private address |
| 422 | `image_url` missing from the body (FastAPI validation) |
| 500 | Inference threw |
| 503 | Model never loaded |

### Run it locally without Docker

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install --index-url https://download.pytorch.org/whl/cpu -r requirements-torch.txt
pip install -r requirements.txt

$env:LOG_FORMAT="text"
uvicorn app.main:app --reload --port 8080
```

Look for `Model loaded` and `Warmup inference complete` in the startup output
before `Uvicorn running`. Those two lines mean the checkpoint deserialized into
the architecture and a forward pass succeeded.

`LOG_FORMAT=text` gives readable logs while developing. Leave it unset in the
container, where JSON is what Cloud Logging wants.

Use a fresh venv here rather than reusing A1's. The point of `requirements.txt`
is that serving needs six packages; installing into an environment that already
has 180 proves nothing.

### Testing

`test_api.py` exercises health, metadata, both prediction paths, four error
cases and the UI in one pass. It runs against any deployment:

```powershell
# local
python test_api.py --image "..\Dan_ML_main\AI_ML_Vanier_A1\data\raw\chest_xray_pneumonia\test\f7ghg9rpnp-1\WhatsApp Image 2021-04-15 at 10.13.14 AM.jpeg"

# deployed
python test_api.py https://xray-api-747277387717.northamerica-northeast1.run.app --image "path\to\xray.jpeg"
```

Expect **15/15**. Without `--image` it falls back to downloading the sample URL
for the upload test, which still passes but exercises one less path.

`SAMPLE_URL` points at `samples/xray-pneumonia.jpeg` in this repo. That choice
matters more than it looks — the two earlier sample URLs both failed, in ways
worth knowing about (see Troubleshooting).

---

## Step 2 — Docker

```powershell
docker build -t xray-api .
docker run --rm -p 8080:8080 xray-api
```

Then <http://localhost:8080>, and in a second terminal:

```powershell
python test_api.py --image "path\to\xray.jpeg"
```

`--rm` deletes the container when you stop it, so dead containers don't pile up.

**Measured result:** **412 MB compressed** (what crosses the network on a pull)
and **1.7 GB on disk** uncompressed. Roughly 130 MB Python slim base, 230 MB
torch and torchvision, 45 MB checkpoint, a few MB of app. Build takes ~106s cold,
seconds when the dependency layer is cached.

Five things in the Dockerfile that matter:

**CPU-only torch.** The default PyPI `torch` wheel bundles CUDA — about 2 GB of
GPU libraries a Cloud Run instance has no card to use. Installing from
`--index-url https://download.pytorch.org/whl/cpu` cuts that to **191.8 MB**, as
the build log shows. It resolves `torch-2.13.0+cpu`, matching the training
environment exactly. The `+cpu` suffix is a PEP 440 local version identifier; a
plain `torch==2.13.0` pin matches it.

**`weights=None` in `model.py`.** A1 built the model with
`models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`, which downloads
ImageNet weights on first call. In a container that means a network fetch at
every cold start and a crash if egress is blocked. Here the architecture is built
empty and every parameter comes from the checkpoint.

**Dependency layers before code.** `requirements*.txt` is copied and installed
before `app/`, so editing `main.py` rebuilds in seconds instead of re-downloading
PyTorch.

**`libgomp1` via apt.** torch links against OpenMP at runtime. Without it the
import fails with `libgomp.so.1: cannot open shared object file`.

**`CMD exec uvicorn ...` in shell form.** Shell form is needed because `${PORT}`
must expand — Cloud Run injects that variable. The `exec` makes uvicorn replace
the shell and become PID 1, so it receives `SIGTERM` directly and shuts down
cleanly when Cloud Run stops the container. Confirmed by the startup log line
`Started server process [1]`.

Docker emits `JSONArgsRecommended` here. It's a static check warning that shell
form breaks signal handling; it doesn't notice the `exec` that fixes exactly
that, so it's safe to ignore.

Building on Apple Silicon? Cloud Run runs amd64, and an arm64 image deploys and
then fails with an exec format error:

```bash
docker build --platform linux/amd64 -t xray-api .
```

`gcloud builds submit` sidesteps this entirely — it builds on Google's amd64
machines.

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
| `BLOCK_PRIVATE_ADDRESSES` | true | See the security note below |

Override at runtime, no rebuild:

```powershell
docker run --rm -p 8080:8080 -e DECISION_THRESHOLD=0.35 -e LOG_FORMAT=text xray-api
```

### One security note

`/predict` fetches URLs on the caller's behalf. Once publicly reachable, that
makes the service a request proxy: anyone can ask it to fetch anything the
container can reach, including cloud metadata endpoints, which on GCP and AWS
hand out credentials to whoever asks from inside the instance. `app/images.py`
resolves the hostname and refuses private, loopback, link-local, reserved and
multicast addresses before making the request, and caps response size and time.
Leave `BLOCK_PRIVATE_ADDRESSES=true` on anything with a public URL.

`test_api.py` covers this — one of its error cases points at `169.254.169.254`,
the metadata address, and expects a 400.

---

## Step 3 — Deploy to Cloud Run

Actual values used for this deployment:

```powershell
$env:PROJECT_ID="project-cea7b8c9-668d-457b-92b"
$env:REGION="northamerica-northeast1"
$env:SERVICE="xray-api"
```

`northamerica-northeast1` is Montréal — lowest latency from here, and it keeps
data in Canada, which is the right default for anything medical-adjacent even in
coursework.

PowerShell uses `$env:VAR="value"` where bash uses `export VAR=value`. All
commands below are PowerShell.

### 3.1 — Authenticate, check billing, enable APIs

```powershell
gcloud auth login
gcloud config set project $env:PROJECT_ID
gcloud billing projects describe $env:PROJECT_ID
```

You want `billingEnabled: true`. Cloud Run's free tier covers a student project
easily, but a billing account still has to be linked or every later command
fails.

```powershell
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
```

A fresh project has all three off by default.

### 3.2 — Grant the build service account its roles

**This step is missing from most tutorials and it cost two failed builds.**

Google no longer auto-grants roles to the default Compute Engine service account,
which is what Cloud Build falls back to. On a new project it arrives with
nothing, and needs three roles:

```powershell
$SA = "747277387717-compute@developer.gserviceaccount.com"

gcloud projects add-iam-policy-binding $env:PROJECT_ID --member="serviceAccount:$SA" --role="roles/storage.objectViewer"
gcloud projects add-iam-policy-binding $env:PROJECT_ID --member="serviceAccount:$SA" --role="roles/artifactregistry.writer"
gcloud projects add-iam-policy-binding $env:PROJECT_ID --member="serviceAccount:$SA" --role="roles/logging.logWriter"
```

What each one prevents:

- **`storage.objectViewer`** — without it, `gcloud builds submit` uploads the
  source tarball and then can't read it back:
  `403 ... does not have storage.objects.get access`. Fails immediately.
- **`artifactregistry.writer`** — without it, the build runs all 14 steps
  successfully and *then* fails at PUSH with
  `denied: Permission 'artifactregistry.repositories.uploadArtifacts' denied`,
  retrying ten times first. A full wasted build, and the error appears hundreds
  of lines after the last thing that went right.
- **`logging.logWriter`** — without it, the build runs but can't write logs and
  fails with a confusing message about log streaming.

Find the service account number for any project with:

```powershell
gcloud projects describe $env:PROJECT_ID --format="value(projectNumber)"
```

IAM changes take up to a minute to propagate. If a command fails immediately
after granting, wait and retry once before assuming the grant didn't work.

### 3.3 — Create the Artifact Registry repo

```powershell
gcloud artifacts repositories create ml-models --repository-format=docker --location=$env:REGION --description="A2 deployment images"
```

Verify:

```powershell
gcloud artifacts repositories list --location=$env:REGION
```

### 3.4 — Build and push

Cloud Build does it server-side — no local Docker daemon, no slow uplink, no
architecture mismatch:

```powershell
gcloud builds submit --tag "$($env:REGION)-docker.pkg.dev/$($env:PROJECT_ID)/ml-models/$($env:SERVICE):v1"
```

**Measured: 3m12s**, uploading 44.5 MiB of context. It respects `.dockerignore`,
so the upload matches a local build's context size.

Watch for `SUCCESS` at the end. On failure, read the log URL it prints rather
than the terminal tail — the real error is usually far above the last line.

<details>
<summary>Or push a locally built image instead</summary>

```powershell
gcloud auth configure-docker "$($env:REGION)-docker.pkg.dev"
docker build --platform linux/amd64 -t "$($env:REGION)-docker.pkg.dev/$($env:PROJECT_ID)/ml-models/$($env:SERVICE):v1" .
docker push "$($env:REGION)-docker.pkg.dev/$($env:PROJECT_ID)/ml-models/$($env:SERVICE):v1"
```
</details>

### 3.5 — Deploy

```powershell
gcloud run deploy $env:SERVICE --image "$($env:REGION)-docker.pkg.dev/$($env:PROJECT_ID)/ml-models/$($env:SERVICE):v1" --region $env:REGION --allow-unauthenticated --port 8080 --memory 2Gi --cpu 1 --timeout 120 --concurrency 4 --min-instances 0 --max-instances 3 --set-env-vars "DECISION_THRESHOLD=0.5,TORCH_NUM_THREADS=1,LOG_LEVEL=INFO"
```

`--allow-unauthenticated` is what makes the endpoint public. Without it every
request needs an IAM token and a grader gets a 403.

Cloud Run starts the container and waits for it to listen on `$PORT`. If it
doesn't within the startup window, the deploy fails and rolls back rather than
leaving a broken revision live. So a successful deploy already proves the model
loaded.

**Why those numbers.** 2 GiB because torch plus the ResNet18 weights sits around
1 GB resident and the default 512 MiB kills the container during load.
Concurrency 4 because inference is CPU-bound with one torch thread — piling more
concurrent requests onto one instance just queues them. Max instances 3 as a cost
ceiling on a student project.

### 3.6 — Verify

```powershell
python test_api.py https://xray-api-747277387717.northamerica-northeast1.run.app --image "path\to\xray.jpeg"
```

15/15, and `P(pneumonia)` matches the local run to four decimals — same weights,
same torch version, same CPU math in a Debian container as in a Windows venv.
That reproducibility is the claim the whole deployment rests on, and it's worth
checking rather than assuming.

### Cold starts

The first request after idle takes **10–20 seconds** while the instance imports
torch and loads 45 MB of weights. Subsequent requests run ~150 ms.

Worth naming as a real tradeoff rather than hiding: `--min-instances 1` removes
the cold start entirely but bills for an always-warm instance. For a graded demo
it's cheaper to hit the URL once shortly before presenting.

### Redeploying

```powershell
gcloud builds submit --tag "$($env:REGION)-docker.pkg.dev/$($env:PROJECT_ID)/ml-models/$($env:SERVICE):v2"
gcloud run deploy $env:SERVICE --image "$($env:REGION)-docker.pkg.dev/$($env:PROJECT_ID)/ml-models/$($env:SERVICE):v2" --region $env:REGION
```

Increment the tag rather than reusing `:v1` or `:latest`. Cloud Run keeps every
revision, so a numbered tag makes rollback to a known-good image trivial.

---

## Step 4 — Web UI

Served at `/` by the same container, so there's no second thing to deploy and no
CORS to configure.

Drop an X-ray on the viewer, click to browse, or paste a URL. The result comes
back as a verdict plus the probability drawn against the decision threshold —
the vertical mark on the gauge is the 0.50 cutoff, so you can see *how close* a
call was rather than only which side it landed on. That's a direct nod to A1's
threshold-tuning work, where the interesting question was never the label but the
margin. Anything within 0.10 of the threshold is flagged as borderline.

The page reads `/metadata` on load, so the marker tracks the server's actual
configured threshold rather than a hard-coded 0.50. Change `DECISION_THRESHOLD`
at deploy time and the gauge moves with it.

The footer states the 60.7% NORMAL recall rather than hiding it. A1's discussion
argued that limitation should be surfaced in any deployment, and a screening tool
that over-flags should say so where its users will actually read it.

---

## Step 5 — Monitoring

**Logs.** Every request produces one structured JSON line with method, path,
status, duration and request ID; predictions add class, probability and inference
time. Cloud Logging parses the `severity` field, so errors show red and every
extra field becomes queryable.

That's the payoff from `logging_config.py`: expanding a `Prediction served` entry
in the console shows `prediction`, `probability_pneumonia` and `inference_ms` as
separate fields, so you can filter for every pneumonia call above 0.9, or every
request slower than 500 ms, without parsing strings.

Health checks log at DEBUG deliberately, so the 30-second `HEALTHCHECK` poll
doesn't bury real traffic.

```powershell
# stream
gcloud run services logs tail $env:SERVICE --region $env:REGION

# recent
gcloud run services logs read $env:SERVICE --region $env:REGION --limit 50

# just the predictions
gcloud logging read 'resource.type="cloud_run_revision" AND jsonPayload.message="Prediction served"' --limit 20 --format json
```

Console: **Cloud Run → xray-api → Logs**.

**Metrics.** **Cloud Run → xray-api → Metrics** — request count, latency
percentiles, instance count, memory and CPU utilisation. They lag two to three
minutes behind real traffic.

Send traffic before screenshotting, or the charts are empty:

```powershell
for ($i=1; $i -le 15; $i++) { python test_api.py https://xray-api-747277387717.northamerica-northeast1.run.app --image "path\to\xray.jpeg" | Out-Null }
```

**Screenshots captured for the deliverable:**

1. Cloud Run service page — URL and green revision
2. Logs tab with `Prediction served` entries expanded
3. Metrics tab after traffic
4. `test_api.py` output against the live URL, 15/15
5. Web UI with a real prediction on screen
6. Artifact Registry showing `xray-api:v1`

---

## Troubleshooting

Everything below was hit while building this.

### Deployment

| Symptom | Cause and fix |
|---|---|
| `403 ... storage.objects.get access` on submit | Build service account missing `roles/storage.objectViewer` — see 3.2 |
| Build runs all 14 steps, then PUSH denied with `uploadArtifacts` | Missing `roles/artifactregistry.writer` — see 3.2 |
| Build fails at `COPY models/...` | Checkpoint not copied in — see Step 0 |
| `exec format error` on deploy | arm64 image on amd64 Cloud Run. Use `gcloud builds submit`, or `--platform linux/amd64` |
| Container fails to start, no useful log | Almost always memory. Raise to 2Gi |
| `libgomp.so.1: cannot open shared object file` | `libgomp1` missing from the apt line |
| `Error loading state_dict` | torch/torchvision drift between training and serving. Pin `requirements-torch.txt` to the A1 versions |
| 403 on the public URL | Missing `--allow-unauthenticated` |
| First request takes 20s, rest fast | Normal cold start |
| `/health` says `model_loaded: false` | Read the startup logs; the loader logs the reason and keeps serving so you can |

### Local and tooling

| Symptom | Cause and fix |
|---|---|
| `/dev/null is an empty file` on container start | uvicorn's `--log-config` can't take `/dev/null` — it parses any non-JSON/YAML path as an ini file and refuses an empty one. Drop the flag; `logging_config.py` already reroutes uvicorn's handlers |
| `ERR_CONNECTION_REFUSED` on localhost:8080 | No port mapping. `docker ps` should show `0.0.0.0:8080->8080/tcp`, not bare `8080/tcp`. `EXPOSE` only documents a port; only `-p` publishes it, and the Docker Desktop Run button leaves it blank unless you expand Optional settings. A mapping can't be added to a running container — stop and recreate |
| `ERR_ADDRESS_INVALID` on 0.0.0.0:8080 | `0.0.0.0` is a bind address ("listen on every interface"), not a destination. Use `localhost:8080` |
| `cd E:\...` fails in Cloud SDK Shell | That's `cmd`, not PowerShell, and bare `cd` won't change drives. `cd /d "E:\..."`, or just use PowerShell |
| `src refspec main does not match any` | No commit exists yet, so `main` points at nothing. `git add .` then `git commit` before `git push` |
| `SyntaxWarning: invalid escape sequence` on a Windows path | `\t` in a Python string is a tab, not two characters. Prefix with `r"..."` — always, for Windows paths |
| Sample URL returns 404 and the file is definitely pushed | Private repo. Raw URLs return a bare 404 to unauthenticated requests with no auth prompt — which the container always is. Make the repo public |
| Sample URL returns 429 | Rate limiting. Wikimedia throttles generic User-Agents and URLs carrying `utm_*` tracking parameters, which miss their CDN cache. Host the sample in your own repo |
| Test count drops by 3 when one thing breaks | `test_predict_url` returns early on a non-200, skipping its remaining checks. One failure, not three |

---

## What was deliberately left out

The A1 `requirements.txt` has ~180 packages: jupyter, matplotlib, seaborn, spacy,
gensim, scikit-learn, kaggle, lime, wordcloud. None of it runs at inference time.

Serving needs a web server, an image decoder and the model runtime — six packages
plus torch. Everything omitted is image size not pulled on every cold start, and
a dependency not inherited.

The same logic explains why the fine-tuned CNN ships rather than the logistic
regression baseline. Serving `baseline_logreg.pkl` would mean carrying a second
frozen ResNet18 to generate 512-dim embeddings before the pickle ever sees the
image — bigger image, two models to load, and a pickle that only unpickles
cleanly against the exact scikit-learn version it was trained with. A1 Step 5
already recommended the CNN on performance (0.822 vs 0.774 macro F1); deployment
cost points the same way.

---

## Possible improvements

What a production version would need that this doesn't have:

- **Weights outside the repo.** 44.8 MB in git works but doesn't scale. Object
  storage or Git LFS, versioned separately from code.
- **Authentication and rate limiting.** The endpoint is fully open. Fine for a
  graded demo, not for anything real.
- **A pinned base image digest.** `python:3.12-slim` is a moving tag; pinning the
  digest makes builds byte-reproducible.
- **Monitoring on prediction distribution.** Request latency is tracked; drift in
  the ratio of PNEUMONIA calls is not, and that's the signal that matters if the
  input distribution shifts.
- **A CI pipeline** running `test_api.py` against a preview revision before
  promoting it to serve traffic.

---

## Credits

Dataset: Kermany, Zhang & Goldbaum, *Labeled Optical Coherence Tomography (OCT)
and Chest X-Ray Images for Classification* (CC BY 4.0), via
[Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
The sample image in `samples/` is redistributed from that dataset under the same
licence.

# Backend setup plan

How to run the three models on a server and get bundles to a JavaScript webapp.

**First, the reassuring part:** the frontend is already pure JavaScript. `webapp/index.html`
has no Python in it and never will — it fetches JSON and PNGs over HTTP and draws with
WebGL. Python exists only inside the pipeline, behind an HTTP API. Nothing about going
live changes the client's language.

---

## 1. What has to run, and how big a machine it needs

| model | checkpoint | VRAM (fp16, est.) | time per image (est.) |
|---|---|---|---|
| SegFormer-B2 | ~110 MB | < 1 GB | ~0.02 s |
| Depth Pro | ~1.9 GB | ~5 GB | ~0.3 s |
| RGB→X (×2 channels) | ~5 GB | ~8 GB | ~3–6 s |
| geometry + emit (CPU) | — | — | ~0.5 s |

**All three resident: roughly 14 GB VRAM. Budget 20 GB+.**

That points at a single **NVIDIA L4 (24 GB)**, which on GCP is the `g2-standard-*`
family. [Around $0.70/hr on-demand](https://gridstackhub.ai/providers/gcp-l4) for one
L4; `g2-standard-8` (8 vCPU, 32 GB RAM, 1×L4) is the sensible shape — the CPU side
matters because the geometry stage and PNG encoding are CPU-bound.

Don't reach for an A100. Nothing here needs 40 GB, and you'd pay 5–6× for latency you
already decided isn't the priority.

**Timing reality:** ~4–7 s per image end to end. That is the async-job design, not a
request/response one. You decided realism over latency, so this is the expected shape,
not a problem to engineer around.

---

## 2. Two architectures — pick by where you are

### A. Single VM (start here)

One `g2-standard-8` running everything: FastAPI, an in-process worker, models loaded
once at startup. Bundles written to GCS.

- Simplest thing that works. One machine to reason about.
- Models stay warm — no cold starts.
- ~$500/month if left running 24/7. Stop it overnight while pre-launch.
- Ceiling is roughly one job at a time, so a handful of concurrent users.

### B. API + GPU workers (when traffic justifies it)

- **Cloud Run** (CPU only) for the API: uploads, job status, signed URLs. Scales to zero.
- **Cloud Tasks** or **Pub/Sub** as the queue.
- **Managed instance group** of L4 VMs as workers, autoscaling on queue depth, min 0
  or 1 depending on how much cold start hurts.

[Cloud Run now supports attaching an L4 directly](https://docs.cloud.google.com/run/docs/configuring/services/gpu)
and scales to zero, which is tempting. The catch is cold start: loading ~7 GB of
weights takes tens of seconds. Viable if you keep one warm instance; painful at true
zero. Worth measuring before committing.

**Recommendation: build A, structure the code so B is a deployment change rather than
a rewrite** — i.e. put the pipeline behind a `process_job(image_bytes) -> bundle` function
that doesn't care whether it was called by an HTTP handler or a queue consumer.

---

## 3. The API

Four endpoints. Async by necessity.

```
POST /v1/jobs
  multipart image upload (cap ~20 MB)
  -> 202 { "job_id": "...", "cached": false }

GET /v1/jobs/{job_id}
  -> { "status": "queued|running|done|failed",
       "progress": "depth|normals|geometry",
       "bundle": { ...signed URLs... },   // when done
       "reason": "..." }                  // when failed

GET /v1/textures            catalogue, served from your existing JSON
GET /healthz                readiness, includes GPU + model-loaded check
```

### Caching by image hash — do this from day one

`sha256(image_bytes)` is the job key. Same photo uploaded twice returns the existing
bundle instantly and never touches the GPU. Users re-trying wallpapers is your dominant
traffic pattern, and it costs zero GPU seconds.

### The bundle response

```jsonc
{
  "scene":   "https://storage.googleapis.com/…/scene.json?X-Goog-Signature=…",
  "maps":    "…/maps.png?…",
  "shading": "…/shading.png?…",
  "photo":   "…/photo.jpg?…"
}
```

Signed URLs, ~1 h expiry. The client fetches directly from GCS — your API never serves
image bytes, which keeps it small and cheap.

### Quality gate

Run `verify.py` before marking a job done. If the scene fails, return
`status: "failed"` with a human-readable `reason` rather than shipping a bad render.
The known failure modes are already enumerated: implausible ceiling height, ambiguous
gravity from a single visible wall, low wall coverage.

**This is the number you don't have yet:** what fraction of *real* uploads fail. Log
every gate outcome from day one — it's the input to whether you need a fallback UX.

---

## 4. GCS layout

```
gs://daga-wallpaper/            (existing — textures)
gs://daga-scenes/               (new — one bundle per image hash)
  bundles/<sha256>/scene.json
                  /maps.png
                  /shading.png
                  /photo.jpg
  uploads/<sha256>.jpg          (optional; see retention below)
```

**Set CORS on `daga-scenes` before you test anything.** Same trap that cost time on the
texture bucket — the client fetches these with `fetch()` and as WebGL textures, both of
which need `Access-Control-Allow-Origin`. Restrict `origin` to your real domains here,
unlike the public texture bucket.

**Retention:** these are photos of people's homes. Decide up front. The defensible
option is to keep the bundle (~1 MB, no recognisable interior once you drop `photo.jpg`)
and delete the original upload after processing. A lifecycle rule on `uploads/` does it.

---

## 5. Frontend deployment

Nothing exotic. The webapp is static files.

- **Firebase Hosting** or **Cloud Storage + Cloud CDN**. Both fine; Firebase is less setup.
- Point `BUNDLES` at your API instead of `../scene/golden`.
- Everything else in `index.html` stays as-is.

The one change: instead of loading a fixed sample, the client uploads a photo, polls
`/v1/jobs/{id}`, then loads the returned bundle URLs into the same code path that
currently reads golden files.

---

## 6. Build order

| step | done when |
|---|---|
| 1. Wrap the pipeline in `process_job(bytes) -> bundle_dir` | callable from a script, no HTTP |
| 2. FastAPI app + in-memory job dict, single VM | curl an image in, get a bundle URL out |
| 3. GCS upload + signed URLs + CORS | the golden-file webapp works against real URLs |
| 4. Hash caching | second upload of the same photo is instant, GPU untouched |
| 5. `verify.py` as an admission gate | a deliberately bad photo returns `failed` with a reason |
| 6. Frontend: upload + poll + render | end to end from a browser |
| 7. Deploy static frontend, lock CORS to your domain | works from a real URL, not localhost |
| 8. Persist jobs (Firestore or Postgres) | survives a VM restart |
| 9. Split to queue + worker MIG | only when concurrency actually hurts |

Steps 1–7 are a working product. 8–9 are hardening.

---

## 7. Environment notes

- **Container the whole thing.** Pin CUDA, torch, and model versions. The pipeline has
  been bitten repeatedly by convention mismatches; a floating dependency stack is the
  same class of problem.
- **Base image:** an NVIDIA CUDA runtime image, then install torch matching it. Don't
  rely on the host driver being what you expect.
- **Bake the weights into the image** rather than downloading at boot. ~7 GB makes the
  image large but startup deterministic and fast.
- **Load models once at process start**, never per request. This is the single biggest
  latency factor after the diffusion model itself.
- **One job at a time per GPU.** Concurrent requests on one L4 will OOM. Serialise with
  a semaphore; queue the rest.

---

## 8. Cost sketch

Rough, and worth re-checking against [GCP's calculator](https://cloud.google.com/products/calculator) —
GPU pricing moves and varies by region.

| item | est. monthly |
|---|---|
| `g2-standard-8` on-demand, 24/7 | ~$500 |
| same, stopped 12 h/day | ~$250 |
| same, Spot | ~$150–200 |
| GCS storage (10k bundles ≈ 10 GB) | ~$0.30 |
| Egress (10k bundle fetches ≈ 10 GB) | ~$1 |
| Cloud Run API (CPU only) | ~$0–5 |

**Storage and bandwidth are noise. The GPU is the entire bill.** Which makes hash
caching and scale-to-zero the only two cost levers that matter.

Per-image marginal cost at ~6 s of L4 time is roughly **$0.0012**. Even at 100k images
that's ~$120 of GPU — the fixed cost of keeping a machine warm dominates until you're
at real volume.

---

## 9. Things to decide before writing code

- **Auth.** Open endpoint invites abuse of a $0.70/hr GPU. At minimum rate-limit by IP;
  better, require a session token from your existing site.
- **Upload limits.** Cap file size and dimensions. Reject non-images before the GPU.
- **What the user sees during 5 s.** Progress states are free to emit (`depth` →
  `normals` → `geometry`) and make the wait feel deliberate rather than broken.
- **What the user sees on failure.** "We couldn't read this room — try a wider shot
  showing more of the walls" is a better answer than a bad render.
- **Retention policy**, published before launch rather than after.

---

## Sources

- [GCP L4 pricing](https://gridstackhub.ai/providers/gcp-l4)
- [Google Cloud GPU instance types and specs](https://www.thundercompute.com/blog/google-cloud-gpu-instances)
- [Cloud Run GPU support](https://docs.cloud.google.com/run/docs/configuring/services/gpu)
- [Cloud Run pricing](https://cloud.google.com/run/pricing)

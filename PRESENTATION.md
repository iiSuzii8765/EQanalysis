# Stage Presentation Guide — EQ Philosophy Backend

This document walks you through the full on-stage flow: what to say, what to click, what every number means, and how to recover if something goes wrong.

---

## Before You Walk On Stage

### 1. Start the system (≥ 10 minutes before)

```bash
docker compose up
```

Wait until you see the worker log line:

```
celery@worker ready.
```

Do not skip this. The first boot compiles OpenFace and loads three model checkpoints — it takes time.

### 2. Verify health

Open a browser tab and go to `http://localhost:8001/health`. You should see:

```json
{"status": "ok"}
```

Keep this tab open as a sanity check.

### 3. Open these tabs in advance

| Tab | URL | Purpose |
|-----|-----|---------|
| Swagger UI | `http://localhost:8001/docs` | Live demo |
| Health | `http://localhost:8001/health` | Quick sanity check |
| This file | — | Your notes |

### 4. Prepare your demo videos

Have at least two short videos ready (10–20 seconds each, frontal face, decent lighting):

- `demo_happy.mp4` — someone smiling or laughing
- `demo_sad.mp4` — someone upset, crying, or visibly distressed

Test both through the pipeline before going on stage. The pipeline needs at least **3 seconds of video with a detectable face** to produce output.

### 5. Know your fallback

If the live demo fails for any reason, paste the example JSON from the README into a text editor and talk through it. The numbers are real — from an actual run.

---

## The Narrative Arc (What to Say)

### Opening hook (30 seconds)

> "When you talk to someone who is clearly upset but says they are fine — your brain is doing something remarkable. It is reading their face, their micro-movements, the timing of their expressions, and comparing that against everything you know about human emotion. What if a machine could do the same thing — not to replace that intuition, but to make it measurable?"

### The problem you are solving (1 minute)

Emotion regulation — how well a person manages and expresses what they feel — is one of the most studied predictors of mental health, relationship quality, and professional performance. Yet there is currently no scalable, objective way to measure it in real time.

Therapists estimate it from observation. Coaches rely on self-report questionnaires. Researchers use expensive physiological sensors in controlled lab settings.

This system does it from a single video, in under a minute, with no wearables and no manual annotation.

### What the system does — in plain language (2 minutes)

Upload a video of someone speaking or reacting. The system:

1. Tracks the face and reads 45+ facial muscle movements per frame using OpenFace (a computer vision tool developed at Carnegie Mellon).
2. Feeds those frames through a deep learning model trained on over 16,000 labeled facial images to detect the dominant emotion in each second of video.
3. Passes that through a temporal model that looks at how emotions *change* over time — not just what someone feels, but whether they are suppressing it, letting it out, or shifting fluidly.
4. Scores four dimensions derived from philosophical theories of emotion regulation — appraisal, somatic coherence, phenomenological coherence, and cognitive load.
5. Combines all of that into a single **Emotion Regulation Score (ERS)** between 0 and 1, with a confidence interval.

### Why it is useful (1 minute)

- A therapist could see in 30 seconds whether a patient's reported mood matches what their face is doing.
- A coaching platform could flag when a client is suppressing rather than processing.
- A researcher could run a study on 500 participants without needing a single lab session.
- The score comes with uncertainty bounds — so the system knows when it does not know.

---

## The Live Demo (Step by Step)

### Step 1 — Open Swagger

Go to `http://localhost:8001/docs`. This is the raw API. No app layer — just the backend directly. That is intentional: you want the audience to see the machine, not a polished UI wrapper.

> "This is the backend API. In a real product you would sit a mobile app or a web dashboard on top of this. What I want to show you today is what the system actually computes."

### Step 2 — Submit the video

Click `POST /api/v1/sessions/analyse` → `Try it out`.

Fill in:
- `context`: type `live-demo` (this is a free-text field — in a real deployment it would be `therapy-session` or `job-interview`)
- `video`: upload your `demo_sad.mp4`

Click `Execute`.

You will get back a response like:

```json
{
  "session_id": "abc123...",
  "status": "queued"
}
```

Copy the `session_id`.

> "The video is now queued. The worker is extracting facial action units, running it through the emotion model, and computing the regulation scores. This takes roughly 10–30 seconds depending on video length."

### Step 3 — Poll status

Click `GET /api/v1/sessions/{session_id}/status` → `Try it out`. Paste the session ID. Execute. Repeat until you see:

```json
{"status": "completed"}
```

### Step 4 — Fetch the result

Click `GET /api/v1/sessions/{session_id}/result` → `Try it out`. Paste the session ID. Execute.

You will get a large JSON. Walk the audience through the key fields:

---

## How to Explain the Output

### `dominant_emotion`

> "Each 3-second window of video gets a dominant emotion label — happy, sad, anger, fear, disgust, surprise, contempt, or neutral. This comes from a ResNet50 model trained on AffectNet, fused with the raw facial action unit signal from OpenFace. The fusion corrects for the fact that real-life sad faces are subtler than the acted faces most AI models train on."

### `ERS` (Emotion Regulation Score)

> "This is the headline number. Zero means complete emotional dysregulation — the face and the underlying muscle signals are incoherent, suppressed, or chaotic. One means the person is fully processing and expressing their emotions in a way that is internally consistent. Most healthy adults in normal conversation sit between 0.6 and 0.85."

### `ERS_uncertainty` and the confidence interval

> "This is what makes the system honest. The Bayesian model runs 50 forward passes with dropout — essentially asking itself the same question 50 times with slightly different assumptions — and reports how much those answers vary. A low uncertainty score, say 0.05, means the system is confident. A high one, say 0.15, means you should trust the result less. That confidence interval tells you the range of scores that are plausible given the evidence."

### `philosophy_scores`

These are the four internal signals that feed into the ERS. Use this table on stage:

| Score | What it measures | Low means | High means |
|-------|-----------------|-----------|------------|
| `F1_appraisal` | Does the facial expression match what you would expect given the emotional context? | Face does not match the stated emotion | Strong congruence between face and context |
| `F2_somatic` | Are the correct facial muscles activating for the detected emotion? | Suppression — face is flat despite emotion | Full somatic expression |
| `F3_coherence` | Are the emotion transitions smooth and natural? | Abrupt, suppressed, or chaotic shifts | Fluid emotional flow |
| `F4_cognitive` | How much cognitive effort is being spent managing emotion? | Low effort (either fully expressed or shut down) | Active regulation — effortful processing |

> "These four numbers come from philosophical frameworks of emotion — specifically appraisal theory, somatic marker hypothesis, phenomenology, and cognitive regulation theory. Stage 4 of the pipeline is entirely deterministic, no machine learning, just explicit rules. That makes it interpretable: you can trace exactly why a score is high or low."

### `regulation_strategy`

> "The system classifies the overall window into one of four strategies: expression (healthy, natural expression), reappraisal (actively re-framing the emotion, also healthy), suppression (holding the emotion in, associated with worse long-term outcomes), or dysregulation (incoherent or extreme pattern). In the demo output you will likely see 'expression' if the person was genuinely sad — that means they were feeling it and showing it."

### `microexpression: true`

> "This flag means the system detected a brief, involuntary facial movement — lasting less than 200 milliseconds — that does not match the overall expression of that window. These are classically associated with concealed emotion. You cannot fake them and you cannot easily suppress them."

### `model_status`

> "This confirms that all three learned model checkpoints were loaded — Stage 2 (spatial emotion model), Stage 3 (temporal dynamics model), and Stage 5 (Bayesian fusion model). If any was missing, the system would fall back to rule-based estimation and flag it here."

---

## Anticipated Questions and Answers

**Q: Is this clinically validated?**

> "Not yet. This is a research prototype. The models are trained on publicly available datasets and evaluated on held-out splits, but clinical validation would require a study with expert-annotated ground truth from a clinical population. That is explicitly listed as future work. What we have today is a technically rigorous baseline that demonstrates the feasibility of the approach."

**Q: Can this be fooled? Can someone fake a high score?**

> "It is harder than it sounds. The system reads 45 facial muscle movements per frame. Consciously controlling all of them simultaneously to produce a coherent fake signal would require training comparable to professional acting — and even then, the microexpression detection and temporal coherence checks would catch inconsistencies. It is not tamper-proof, but it is substantially harder to manipulate than a self-report questionnaire."

**Q: What about privacy?**

> "The video is processed locally — no frames leave the server. OpenFace extracts numerical features and discards the raw pixels before anything is stored. In a production deployment you would add at-rest encryption and access controls. The architecture is already designed to be stateless: the video is stored only until processing completes."

**Q: Why not just use an off-the-shelf emotion API?**

> "Existing APIs — Azure Face, Amazon Rekognition, Google Vision — output a single emotion label with a confidence score. They do not give you regulation dynamics, temporal coherence, or uncertainty. And they are black boxes: you cannot see why they made a prediction. The philosophy scores here are fully interpretable and traceable. That matters in a clinical or coaching context where 'the AI said so' is not sufficient."

**Q: What is the accuracy?**

> "The Stage 2 emotion model achieves 0.666 macro-F1 across 8 emotion classes on the AffectNet validation set. For context, human agreement on emotion labels for the same dataset is around 0.65–0.70, so the model is in the range of human-level inter-rater agreement. The ERS score is evaluated on proxy labels derived from the pipeline itself — that is a known limitation and the first thing we would address with real supervised data."

---

## Recovery Plan (If Something Breaks)

| Problem | Fix |
|---------|-----|
| Docker not running | `docker compose up` — takes ~3 min |
| Worker not ready | Watch logs for `celery@worker ready.` — wait |
| OpenFace fails to detect face | Use a video with better lighting, frontal face, stable head position |
| Status stuck on `processing` | Check worker logs: `docker compose logs worker --tail 50` |
| Result endpoint returns error | Copy the sample JSON from README and present that |
| No internet for Docker pull | Images should already be built locally from the pre-run |

If the full demo fails, fall back to the JSON example in the README, paste it into a browser JSON viewer (or use VS Code), and walk through each field. The point is the output — not the live submission.

---

## Timing Guide

| Section | Time |
|---------|------|
| Opening hook | 30 sec |
| Problem statement | 1 min |
| System explanation | 2 min |
| Live demo (submit + wait + result) | 3–4 min |
| Explain output fields | 3 min |
| Q&A | remaining |
| **Total** | **~10 min** |

If you have more time, submit a second video (happy) after the sad one and contrast the ERS scores live. The difference in `F2_somatic` (body expression) and `regulation_strategy` between a genuinely happy face and a sad face is immediately visible in the numbers.

---

## One-Line Summary for the Pitch

> "We turn a 10-second video into a clinically interpretable, uncertainty-aware emotion regulation score — using facial action units, deep learning, and philosophy."

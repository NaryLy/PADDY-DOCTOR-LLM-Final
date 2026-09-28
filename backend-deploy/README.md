# Rice Disease Doctor -- backend API

Deployed on Render as a Docker web service (root directory: `backend-deploy`).

FastAPI backend serving the fine-tuned ResNet18 rice disease classifier
(88.7% test accuracy, 10 classes). Deployed here so the static frontend
(hosted on Netlify) has a live API to call.

Endpoints: `GET /api/health`, `POST /api/predict`, `GET /api/history`,
`DELETE /api/history`.

Full project (training code, dataset prep, report):
https://github.com/NaryLy/PADDY-DOCTOR-LLM-Final

Note: prediction history and uploaded images live on this Space's ephemeral
filesystem and reset whenever the Space restarts or goes to sleep -- this is
a course-project demo API, not persistent storage.

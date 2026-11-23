# Put the dashboard online (free link you can open from any browser)

`localhost` only works on your own laptop while a terminal is running. To get a **public link**
(like `https://your-app.streamlit.app`) deploy the dashboard on **Streamlit Community Cloud** (free).

## Step 1 - Push the project to GitHub
1. Install Git: https://git-scm.com/download/win
2. Create a new empty repository on https://github.com/new (name: `ecommerce-purchase-intent-xai`, **do not** add README).
3. In the project folder (open `cmd` there) run:
```
git init
git add .
git commit -m "Purchase intent prediction with explainable AI"
git branch -M main
git remote add origin https://github.com/<YOUR-USERNAME>/ecommerce-purchase-intent-xai.git
git push -u origin main
```
(A browser window may ask you to sign in to GitHub - allow it.)

All files are small (largest is ~0.3 MB), so nothing needs special handling.

## Step 2 - Deploy
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app** -> **Deploy a public app from GitHub**.
3. Fill in:
   - Repository: `<YOUR-USERNAME>/ecommerce-purchase-intent-xai`
   - Branch: `main`
   - Main file path: `dashboard/app.py`
4. Open **Advanced settings** and choose **Python 3.12** if the option is shown.
5. Click **Deploy**. The first build takes a few minutes (it installs `dashboard/requirements.txt`).
6. You get a link like `https://<something>.streamlit.app` - paste it in Chrome. Put the link in your README and the portal submission.

## Notes
- The app reads `models/`, `reports/` and `dataset/` from the repo, so they must stay committed.
- Free apps go to sleep after some inactivity; opening the link wakes them up (takes ~30 seconds).
- Only the **dashboard** is deployed this way. The FastAPI service needs a separate host (e.g. Render / AWS - see `deploy/aws/`).
- If the build fails, open **Manage app -> Logs** on Streamlit Cloud and read the last red lines.

# Final submission checklist

## A. Put the code on GitHub (no command line needed)
**Option 1 - GitHub Desktop (recommended):**
1. Install GitHub Desktop (desktop.github.com) and sign in with your GitHub account.
2. File -> Add local repository -> choose the project folder (the one that contains `README.md`).
   If it says "not a Git repository", click **create a repository**.
3. Write a summary like `Initial commit: purchase intent XAI project` -> **Commit to main**.
4. Click **Publish repository**, name it `ecommerce-purchase-intent-xai`, keep "Keep this code private" **unchecked**
   (Streamlit free hosting needs a public repo unless you link a private one) -> **Publish**.

**Option 2 - Browser upload:** github.com -> New repository -> **uploading an existing file** -> open the project folder,
select **everything inside it** (Ctrl+A, not the folder itself) and drag into the page -> Commit changes.
(The project has 83 files, below the 100-files-per-upload limit.)

## B. Get a public link (opens in any browser, no cmd)
1. share.streamlit.io -> sign in with GitHub -> **Create app**.
2. Repository: your repo - Branch: `main` - Main file path: `dashboard/app.py`.
3. Advanced settings -> Python **3.12** (if shown) -> **Deploy** (first build takes a few minutes).
4. Copy the `https://<name>.streamlit.app` link. Paste it in the README (pencil icon on GitHub) and in the portal.

## C. Before you submit
- [ ] Dashboard opens from the public link; Live prediction works (try PageValues 35 and 0).
- [ ] README shows your GitHub + dashboard links.
- [ ] Demo video recorded (`docs/demo_script.md`) and uploaded (Drive/YouTube unlisted) - link in portal.
- [ ] You can explain: why PR-AUC, why SMOTE is inside the pipeline, the PageValues issue (`docs/VIVA_PREP.md`).
- [ ] Portal: mark **Work started**, then **Submit for review** with the repo link (and video / dashboard link) before **1 Feb 2027**.

# Student Subject-Wise Academic Risk Prediction — Streamlit

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Run:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
4. Streamlit will open the app in your browser.

The app recreates the synthetic 300-student dataset using the generation logic from the project notebook, so no separate CSV is required. You can download the CSV from **Dataset Explorer**.

## Deploy to Streamlit Community Cloud
1. Create a GitHub repository and upload `app.py` and `requirements.txt`.
2. Visit https://share.streamlit.io/ and sign in with GitHub.
3. Select **Create app**, choose your repository and branch, and set the main file path to `app.py`.
4. Click **Deploy** and wait for the app to build.

## Important note
The dataset is synthetic and its target labels are generated from rules that use related input features. High test metrics do not establish real-world performance. Do not use predictions to make consequential decisions about real students without independent validation and appropriate oversight.

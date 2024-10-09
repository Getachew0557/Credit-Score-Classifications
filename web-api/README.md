## Deploying the Web API on Render

Render is a cloud platform that allows you to host web apps and APIs easily. Here's a step-by-step guide on how to deploy your Flask app to Render using your GitHub repository:

1. Create a Render Account
    - Go to Render and sign up for a free account (if you don’t have one yet).
    - After signing up, you will land on the Render dashboard.

2. Connect Your GitHub Repository
    - From the Render dashboard, click on New + and select Web Service.
    - You'll be prompted to connect your GitHub account. Authorize Render to access your repositories.
    - After connecting, you'll be able to see your GitHub repositories. Select the repository containing your project (i.e., Credit-Score-Classifications).

3. Configure the Web Service Settings
Once you've selected your repository, configure the following settings:

    - **Name**: Give your web service a name, such as credit-score-api.

    - **Environment**: Choose Python 3.

    - **Build Command**: If you're using a requirements.txt file to manage dependencies, set the build command to:

    ```bash
    pip install -r requirements.txt
    ```
    - **Start Command**: This will be the command Render uses to start your Flask application. Set it to:

    ```bash
    python flask-app/app.py
    ```
    If you're using Gunicorn (a production server for Flask apps), you can set this to:

    ```bash
    gunicorn flask-app.app:app
    ```
    - **Region**: Choose the region closest to your primary users.

    - **Branch**: Choose the branch you want to deploy from (e.g., main).

4. Set Up Environment Variables
If your Flask app requires environment variables (e.g., FLASK_ENV, SECRET_KEY), add them under the Environment section.

To add them:

    - Scroll down to the Environment Variables section.
    - Add variables like PORT=5000, FLASK_ENV=production, or any other necessary credentials for databases or third-party APIs.

5. Deploy the Web App
After configuring your settings, click Create Web Service. Render will automatically:

    - Clone your GitHub repository.
    - Install dependencies based on requirements.txt.
    - Run the Start Command to deploy your web app.

6. Access the Deployed App
Once deployment is complete, you'll receive a URL to access your live Flask application (e.g., https://credit-score-api.onrender.com).

7. Test the Deployed Web App
Open the URL in your browser. You should see your application interface, and you can fill out the form to submit and receive predictions.
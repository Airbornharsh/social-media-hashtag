// server.ts

import express, { Request, Response } from "express";
import bodyParser from "body-parser";
import { Options, PythonShell } from "python-shell";
import axios from "axios";

const app = express();
const PORT = 3000;

// Get path to Conda environment's Python executable
// const condaEnvName = "/home/airbornharsh/miniconda3/envs/myenv"; // Replace with your Conda environment name
// const pythonPath = execSync(`conda run -n ${condaEnvName} which python`)
//   .toString()
//   .trim();
const pythonPath =
  "/home/airbornharsh/Programming/internship/freelancer/social-media-hashtag-username/model_training/env/bin/python";

// Configure body parser middleware
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// Define POST endpoint for predicting hashtags
app.post("/predict", async (req: Request, res: Response) => {
  try {
    // Extract latitude and longitude from the request body
    const { latitude, longitude } = req.body;

    // Prepare options for PythonShell
    const options = {
      mode: "text",
      pythonOptions: ["-W", "ignore"],
      pythonPath, // Use the Python executable from the Conda environment
      scriptPath:
        "/home/airbornharsh/Programming/internship/freelancer/social-media-hashtag-username/model_training/scripts", // Path to the directory containing the Python script
      args: [latitude, longitude], // Pass latitude and longitude as arguments to the Python script
    };

    const data = await PythonShell.run(
      "predict_hashtags.py",
      options as Options
    );
    console.log(data);
    res.json({ hashtags: data });
  } catch (e: any) {
    console.log(e);
    res.json({ error: e.message });
  }
});

app.get("/hashtag", async (req: Request, res: Response) => {
  try {
  } catch (e: any) {
    console.log(e);
    res.json({ error: e.message });
  }
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});

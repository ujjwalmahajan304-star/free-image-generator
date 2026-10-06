const express = require("express");
const multer = require("multer");
const { InferenceClient } = require("@huggingface/inference");

const app = express();
const upload = multer({ storage: multer.memoryStorage() });

app.use(express.static("public"));

const client = new InferenceClient(process.env.HF_TOKEN);

app.post("/api/generate", upload.single("image"), async (req, res) => {
  try {
    if (!process.env.HF_TOKEN) {
      return res.status(500).send("HF_TOKEN is not configured on the server.");
    }

    const prompt = req.body.prompt || "";
    let image;

    if (req.file) {
      image = await client.imageToImage({
        provider: "fal-ai",
        model: "black-forest-labs/FLUX.2-klein-9B",
        inputs: req.file.buffer,
        prompt: prompt
      });
    } else {
      image = await client.textToImage({
        provider: "fal-ai",
        model: "black-forest-labs/FLUX.1-schnell",
        inputs: prompt
      });
    }

    res.type("png").send(Buffer.from(await image.arrayBuffer()));
  } catch (e) {
    console.error(e);
    res.status(500).send(e.message || "AI generation error");
  }
});

app.listen(process.env.PORT || 3000, () => {
  console.log("Free Image Generator running");
});

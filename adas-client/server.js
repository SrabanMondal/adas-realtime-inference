const https = require("https");
const fs = require("fs");
const next = require("next");

const dev = true;
const app = next({ dev });
const handle = app.getRequestHandler();

app.prepare().then(() => {
  https.createServer(
    {
      key: fs.readFileSync("key.pem"),
      cert: fs.readFileSync("cert.pem"),
    },
    (req, res) => handle(req, res)
  ).listen(3000, "0.0.0.0", () => {
    console.log("Next.js HTTPS running on https://192.168.1.7:3000");
  });
});

const config = require('../config');

/**
 * Runs after multer has parsed the multipart upload (req.file). Rejects anything
 * that isn't an expected image type/size *before* it gets forwarded to the ML
 * backend — the gateway is the choke point, so this is where we stop bad input,
 * not inside the model-serving code.
 */
function validateXrayUpload(req, res, next) {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded. Expected multipart field "file".' });
  }

  if (!config.upload.allowedMimeTypes.includes(req.file.mimetype)) {
    return res.status(415).json({
      error: `Unsupported file type "${req.file.mimetype}". Allowed: ${config.upload.allowedMimeTypes.join(', ')}`,
    });
  }

  if (req.file.size > config.upload.maxSizeBytes) {
    return res.status(413).json({
      error: `File too large. Max size is ${config.upload.maxSizeBytes / (1024 * 1024)}MB.`,
    });
  }

  // Basic magic-byte sanity check for JPEG/PNG so a renamed .exe with a spoofed
  // mimetype header doesn't sail through on the Content-Type field alone.
  const buffer = req.file.buffer;
  const isPng = buffer.slice(0, 8).toString('hex') === '89504e470d0a1a0a';
  const isJpeg = buffer.slice(0, 3).toString('hex') === 'ffd8ff';
  if ((req.file.mimetype === 'image/png' && !isPng) ||
      (req.file.mimetype === 'image/jpeg' && !isJpeg)) {
    return res.status(415).json({ error: 'File contents do not match declared image type.' });
  }

  next();
}

module.exports = { validateXrayUpload };

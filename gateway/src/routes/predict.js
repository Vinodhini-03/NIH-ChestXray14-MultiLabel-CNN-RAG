const express = require('express');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');
const config = require('../config');
const { requireAuth } = require('../middleware/auth');
const { validateXrayUpload } = require('../middleware/validateUpload');
const logger = require('../utils/logger');

const router = express.Router();

// Keep the file in memory (not on disk) — it's forwarded straight to FastAPI and
// discarded, so there's no need to write PHI-adjacent data to the filesystem.
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: config.upload.maxSizeBytes },
});

router.post('/', requireAuth, upload.single('file'), validateXrayUpload, async (req, res, next) => {
  try {
    const form = new FormData();
    form.append('file', req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype,
    });

    const response = await axios.post(`${config.fastApiBaseUrl}/predict`, form, {
      headers: form.getHeaders(),
      maxContentLength: config.upload.maxSizeBytes * 2,
      maxBodyLength: config.upload.maxSizeBytes * 2,
      timeout: 30_000,
    });

    logger.info('Prediction served', { email: req.user.email });
    return res.json(response.data);
  } catch (err) {
    if (err.response) {
      // FastAPI responded with an error — pass its status through, but not its raw body,
      // in case it contains internal details we don't want to expose to the client.
      logger.error('FastAPI /predict returned an error', {
        status: err.response.status,
        data: err.response.data,
      });
      return res.status(err.response.status >= 500 ? 502 : err.response.status).json({
        error: 'The prediction service could not process this image.',
      });
    }
    if (err.code === 'ECONNREFUSED' || err.code === 'ECONNABORTED') {
      return res.status(503).json({ error: 'Prediction service is unavailable. Try again shortly.' });
    }
    return next(err);
  }
});

module.exports = router;

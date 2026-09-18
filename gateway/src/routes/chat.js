const express = require('express');
const axios = require('axios');
const config = require('../config');
const { requireAuth } = require('../middleware/auth');
const logger = require('../utils/logger');

const router = express.Router();

const MAX_MESSAGE_LENGTH = 2000;

router.post('/stream', requireAuth, async (req, res, next) => {
  const { question, probs } = req.body;

  if (typeof question !== 'string' || question.trim().length === 0) {
    return res.status(400).json({ error: 'A non-empty "question" string is required.' });
  }
  if (question.length > MAX_MESSAGE_LENGTH) {
    return res.status(413).json({ error: `Question exceeds ${MAX_MESSAGE_LENGTH} characters.` });
  }
  if (probs !== undefined && !Array.isArray(probs)) {
    return res.status(400).json({ error: '"probs" must be an array if provided.' });
  }

  try {
    const upstream = await axios.post(
      `${config.fastApiBaseUrl}/chat/stream`,
      { question, probs: probs || new Array(14).fill(0) },
      { responseType: 'stream', timeout: 60_000 }
    );

    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    upstream.data.pipe(res);

    upstream.data.on('error', (streamErr) => {
      logger.error('Upstream stream error', { message: streamErr.message });
      res.end();
    });

    req.on('close', () => {
      upstream.data.destroy();
    });
  } catch (err) {
    if (err.response) {
      return res.status(err.response.status >= 500 ? 502 : err.response.status).json({
        error: 'The chat service could not process this request.',
      });
    }
    if (err.code === 'ECONNREFUSED' || err.code === 'ECONNABORTED') {
      return res.status(503).json({ error: 'Chat service is unavailable. Try again shortly.' });
    }
    return next(err);
  }
});

module.exports = router;
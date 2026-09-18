const express = require('express');
const helmet = require('helmet');
const cors = require('cors');
const morgan = require('morgan');
const rateLimit = require('express-rate-limit');

const config = require('./config');
const logger = require('./utils/logger');
const { seedDemoUser } = require('./utils/userStore');
const errorHandler = require('./middleware/errorHandler');

const authRoutes = require('./routes/auth');
const predictRoutes = require('./routes/predict');
const chatRoutes = require('./routes/chat');

const app = express();

// --- Security headers ---
app.use(helmet());

// --- CORS: only the configured React app origins may call this API ---
app.use(
  cors({
    origin: (origin, callback) => {
      // Allow same-origin/non-browser requests (no Origin header) and configured origins.
      if (!origin || config.allowedOrigins.includes(origin)) {
        return callback(null, true);
      }
      logger.warn('Blocked CORS request', { origin });
      return callback(new Error('Not allowed by CORS'));
    },
    credentials: true,
  })
);

app.use(express.json({ limit: '1mb' }));
app.use(morgan(config.nodeEnv === 'production' ? 'combined' : 'dev'));

// --- Rate limiting ---
// General limiter for everything...
app.use(
  rateLimit({
    windowMs: config.rateLimit.windowMs,
    max: config.rateLimit.maxRequests,
    standardHeaders: true,
    legacyHeaders: false,
    message: { error: 'Too many requests. Please slow down.' },
  })
);

// ...and a tighter one specifically for /predict, since each call triggers a
// full model inference pass and is the most expensive route to abuse.
const predictLimiter = rateLimit({
  windowMs: config.rateLimit.windowMs,
  max: config.rateLimit.predictMax,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many prediction requests. Please slow down.' },
});

app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'chestai-gateway', time: new Date().toISOString() });
});

app.use('/auth', authRoutes);
app.use('/predict', predictLimiter, predictRoutes);
app.use('/chat', chatRoutes);

app.use((req, res) => {
  res.status(404).json({ error: 'Not found' });
});

app.use(errorHandler);

async function start() {
  await seedDemoUser(); // demo@chestai.dev / ChangeMe123! — remove once real signup is wired to a DB
  app.listen(config.port, () => {
    logger.info(`chestai-gateway listening on port ${config.port}`, {
      env: config.nodeEnv,
      fastApiBaseUrl: config.fastApiBaseUrl,
    });
  });
}

start();

module.exports = app;

require('dotenv').config();

function requireEnv(name, fallback) {
  const value = process.env[name] ?? fallback;
  if (value === undefined) {
    throw new Error(`Missing required environment variable: ${name}`);
  }
  return value;
}

const config = {
  port: parseInt(process.env.PORT || '4000', 10),
  nodeEnv: process.env.NODE_ENV || 'development',
  fastApiBaseUrl: requireEnv('FASTAPI_BASE_URL', 'http://localhost:8000'),
  jwt: {
    secret: requireEnv('JWT_SECRET', 'dev_only_insecure_secret_change_me'),
    expiresIn: process.env.JWT_EXPIRES_IN || '1h',
  },
  allowedOrigins: (process.env.ALLOWED_ORIGINS || 'http://localhost:5173')
    .split(',')
    .map((origin) => origin.trim())
    .filter(Boolean),
  rateLimit: {
    windowMs: parseInt(process.env.RATE_LIMIT_WINDOW_MS || '900000', 10),
    maxRequests: parseInt(process.env.RATE_LIMIT_MAX_REQUESTS || '100', 10),
    predictMax: parseInt(process.env.PREDICT_RATE_LIMIT_MAX || '20', 10),
  },
  upload: {
    maxSizeBytes: parseInt(process.env.MAX_UPLOAD_SIZE_MB || '10', 10) * 1024 * 1024,
    allowedMimeTypes: ['image/jpeg', 'image/png', 'image/dicom', 'application/dicom'],
  },
};

if (config.nodeEnv === 'production' && config.jwt.secret === 'dev_only_insecure_secret_change_me') {
  throw new Error('Refusing to start in production with the default JWT_SECRET. Set a real secret.');
}

module.exports = config;

const logger = require('../utils/logger');

// eslint-disable-next-line no-unused-vars
function errorHandler(err, req, res, next) {
  const status = err.status || 500;

  logger.error('Unhandled error', {
    message: err.message,
    path: req.path,
    method: req.method,
    status,
  });

  // Never leak stack traces or internal details to the client.
  res.status(status).json({
    error: status === 500 ? 'Internal server error' : err.message,
  });
}

module.exports = errorHandler;

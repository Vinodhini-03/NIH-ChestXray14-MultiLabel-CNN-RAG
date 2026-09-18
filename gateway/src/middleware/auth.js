const jwt = require('jsonwebtoken');
const config = require('../config');
const logger = require('../utils/logger');

/**
 * Verifies the Bearer JWT on the request and attaches { email, role } to req.user.
 * Rejects missing/invalid/expired tokens with 401 rather than leaking why (avoid
 * giving attackers a token-validity oracle).
 */
function requireAuth(req, res, next) {
  const header = req.headers.authorization || '';
  const [scheme, token] = header.split(' ');

  if (scheme !== 'Bearer' || !token) {
    return res.status(401).json({ error: 'Missing or malformed Authorization header' });
  }

  try {
    const payload = jwt.verify(token, config.jwt.secret);
    req.user = { email: payload.sub, role: payload.role };
    return next();
  } catch (err) {
    logger.warn('JWT verification failed', { reason: err.message });
    return res.status(401).json({ error: 'Invalid or expired token' });
  }
}

/**
 * Role-based permission gate. Use after requireAuth.
 * e.g. router.get('/admin/stats', requireAuth, requireRole('admin'), handler)
 */
function requireRole(...allowedRoles) {
  return (req, res, next) => {
    if (!req.user) {
      return res.status(401).json({ error: 'Not authenticated' });
    }
    if (!allowedRoles.includes(req.user.role)) {
      logger.warn('Permission denied', { email: req.user.email, role: req.user.role, allowedRoles });
      return res.status(403).json({ error: 'You do not have permission to perform this action' });
    }
    return next();
  };
}

module.exports = { requireAuth, requireRole };

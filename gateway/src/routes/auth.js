const express = require('express');
const jwt = require('jsonwebtoken');
const config = require('../config');
const { findUser, createUser, verifyPassword } = require('../utils/userStore');
const logger = require('../utils/logger');

const router = express.Router();

function isValidEmail(email) {
  return typeof email === 'string' && /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}

function isValidPassword(password) {
  return typeof password === 'string' && password.length >= 8;
}

function issueToken(user) {
  return jwt.sign({ sub: user.email, role: user.role }, config.jwt.secret, {
    expiresIn: config.jwt.expiresIn,
  });
}

router.post('/register', async (req, res, next) => {
  try {
    const { email, password } = req.body;

    if (!isValidEmail(email) || !isValidPassword(password)) {
      return res.status(400).json({ error: 'Valid email and password (min 8 chars) are required.' });
    }

    const user = await createUser(email, password);
    const token = issueToken(user);
    logger.info('User registered', { email });
    return res.status(201).json({ token, role: user.role });
  } catch (err) {
    if (err.message === 'User already exists') {
      return res.status(409).json({ error: 'An account with this email already exists.' });
    }
    return next(err);
  }
});

router.post('/login', async (req, res, next) => {
  try {
    const { email, password } = req.body;

    if (!isValidEmail(email) || typeof password !== 'string') {
      return res.status(400).json({ error: 'Email and password are required.' });
    }

    const user = await findUser(email);
    // Compare against a dummy hash even when the user doesn't exist, so login
    // takes the same time either way and doesn't leak which emails are registered.
    const passwordOk = user
      ? await verifyPassword(user, password)
      : await verifyPassword({ passwordHash: '$2a$10$invalidsaltinvalidsaltinvalidsalt' }, password).catch(() => false);

    if (!user || !passwordOk) {
      return res.status(401).json({ error: 'Invalid email or password.' });
    }

    const token = issueToken(user);
    return res.json({ token, role: user.role });
  } catch (err) {
    return next(err);
  }
});

module.exports = router;

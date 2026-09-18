const bcrypt = require('bcryptjs');

// Demo in-memory store so the gateway is runnable without provisioning a database.
// Replace with a real users table (Postgres/Mongo) before shipping to production —
// passwords are hashed here so the pattern is correct even though storage isn't durable.
const users = new Map();

async function seedDemoUser() {
  if (users.has('demo@chestai.dev')) return;
  const passwordHash = await bcrypt.hash('ChangeMe123!', 10);
  users.set('demo@chestai.dev', {
    email: 'demo@chestai.dev',
    passwordHash,
    role: 'clinician', // roles: 'clinician' can call /predict and /chat; 'admin' can also view usage stats
  });
}

async function findUser(email) {
  return users.get(email) || null;
}

async function createUser(email, password, role = 'clinician') {
  if (users.has(email)) {
    throw new Error('User already exists');
  }
  const passwordHash = await bcrypt.hash(password, 10);
  const user = { email, passwordHash, role };
  users.set(email, user);
  return user;
}

async function verifyPassword(user, password) {
  return bcrypt.compare(password, user.passwordHash);
}

module.exports = { seedDemoUser, findUser, createUser, verifyPassword };

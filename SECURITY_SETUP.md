# 🔐 SECURE ENVIRONMENT SETUP

## ⚠️ IMPORTANT SECURITY NOTICE
**NEVER commit API keys or secrets to version control!**

## 🔑 Setting Up Your OpenAI API Key for Render:

### Method 1: From your .env file (Recommended)
1. Check your local `.env` file for the API key
2. Copy the value after `OPENAI_API_KEY=`
3. In Render dashboard, add it as an environment variable

### Method 2: Get a new key from OpenAI
1. Go to https://platform.openai.com/api-keys
2. Create a new API key
3. Copy the key (starts with `sk-proj-...`)
4. Add it to Render environment variables

## 🛡️ Security Best Practices:
- ✅ API keys go in environment variables only
- ✅ Never commit `.env` files
- ✅ Use different keys for development/production
- ✅ Regenerate keys if exposed
- ✅ Monitor API usage regularly

## 📝 Environment Variables for Render:
```
OPENAI_API_KEY=your-actual-api-key-here
OPENAI_MODEL=gpt-4
DEBUG=False
```

## 🚨 If You Accidentally Commit a Secret:
1. Immediately regenerate the API key
2. Remove it from the commit history
3. Update all services using the old key

Stay secure! 🔐

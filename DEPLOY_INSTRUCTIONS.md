# 🚀 Deploy Updates to Render

## Quick Deploy Guide

Your Render deployment is already set up. Just push your changes to update the website!

---

## 📋 Pre-Deployment Checklist

✅ Model retrained with realistic ZAMCELCO data  
✅ Stats cache updated  
✅ Old files cleaned up  
✅ Data period: March-April 2024 (1.45 months)  
✅ Model accuracy: 97.32%  
✅ All essential files present  

---

## 🔄 Deploy Steps

### Step 1: Check Git Status
```bash
git status
```

### Step 2: Add All Changes
```bash
git add .
```

### Step 3: Commit Changes
```bash
git commit -m "Update to ZAMCELCO realistic data (March-April 2024) - 97.32% accuracy"
```

### Step 4: Push to GitHub
```bash
git push origin main
```
*Note: Replace `main` with `master` if that's your branch name*

---

## ⚡ What Happens Next

1. **GitHub receives your push** ✓
2. **Render detects the changes** (automatic)
3. **Render starts building** (~2-3 minutes)
   - Installs dependencies from `requirements.txt`
   - Loads updated model and data
4. **Render deploys** (automatic)
5. **Your website updates** ✓

---

## 📊 What's Being Updated

### New Features:
- ✅ Realistic ZAMCELCO smart meter data
- ✅ Zamboanga City climate adjustments
- ✅ Power brownout handling
- ✅ Missing values and sensor noise
- ✅ 1.45 months of data (March-April 2024)
- ✅ 97.32% model accuracy

### Files Updated:
- `smart_meter_data.csv` - New realistic ZAMCELCO data
- `electricity_model.pkl` - Retrained model
- `stats_cache.json` - Updated statistics
- `train_model.py` - Handles missing values
- `README.md` - Complete documentation
- `ZAMCELCO_DATA_SUMMARY.md` - Data documentation

### Files Removed:
- Old backup files (1.86 MB freed)
- Unused analysis scripts
- Redundant documentation

---

## 🔍 Monitor Deployment

### Check Render Dashboard:
1. Go to https://dashboard.render.com
2. Select your service
3. Click "Events" tab
4. Watch the deployment progress

### Deployment Logs:
- Build logs show installation progress
- Deploy logs show startup messages
- Look for: "Model loaded: RandomForestRegressor"

---

## ✅ Verify Deployment

Once deployed, test your website:

### 1. Health Check
```
https://your-app.onrender.com/health
```
Should return:
```json
{
  "status": "ok",
  "model": "RandomForestRegressor",
  "stats_loaded": true
}
```

### 2. Homepage
```
https://your-app.onrender.com/
```
Should show:
- Updated model accuracy (97.32%)
- ZAMCELCO branding
- March-April 2024 data period

### 3. Make a Test Prediction
- Select any dorm and room
- Enter environmental conditions
- Check if prediction works
- Verify cost calculation

---

## 🐛 Troubleshooting

### If Build Fails:
```bash
# Check requirements.txt is present
cat requirements.txt

# Verify all files are committed
git status
```

### If Model Doesn't Load:
- Check `electricity_model.pkl` is in repository
- Verify file size: ~3.3 MB
- Check Render logs for error messages

### If Stats Don't Show:
- Verify `stats_cache.json` is present
- Check file size: ~130 KB
- Ensure it's committed to git

---

## 📝 Commit Message Templates

### For this update:
```bash
git commit -m "Update to ZAMCELCO realistic data (March-April 2024) - 97.32% accuracy"
```

### For future updates:
```bash
# Model improvements
git commit -m "Improve model accuracy to XX%"

# Data updates
git commit -m "Add new data from [date] to [date]"

# Bug fixes
git commit -m "Fix: [description of bug]"

# Feature additions
git commit -m "Add: [new feature description]"
```

---

## 🎯 Expected Results

After deployment, your website will show:

### Updated Dashboard:
- Model Type: RandomForestRegressor
- MAE: 0.0199
- RMSE: 0.0402
- R²: 0.9732 (97.32%)
- CV R²: 0.9697

### Updated Data Info:
- Data Source: ZAMCELCO Smart Meter
- Location: Zamboanga City
- Period: March 1 - April 14, 2024
- Records: 2,089

### New Features:
- Handles missing sensor values
- Accounts for power brownouts
- Zamboanga climate adjustments
- Realistic sensor imperfections

---

## ⏱️ Deployment Timeline

| Step | Duration | Status |
|------|----------|--------|
| Git push | ~5 seconds | Instant |
| Render detects | ~10 seconds | Automatic |
| Build starts | ~30 seconds | Automatic |
| Install dependencies | ~1-2 minutes | Automatic |
| Deploy | ~30 seconds | Automatic |
| **Total** | **~2-3 minutes** | ✅ |

---

## 🔄 Auto-Deploy Settings

Your Render service should have:
- ✅ Auto-deploy enabled
- ✅ Branch: main (or master)
- ✅ Build command: `pip install -r requirements.txt`
- ✅ Start command: `gunicorn app:app --timeout 120 --workers 1 --threads 2`

---

## 📞 Need Help?

### Check Render Logs:
```
Dashboard → Your Service → Logs
```

### Common Issues:
1. **Build timeout** - Increase timeout in Render settings
2. **Memory error** - Upgrade to paid plan or reduce model size
3. **Module not found** - Check requirements.txt

---

## ✅ Post-Deployment Checklist

After deployment completes:

- [ ] Visit your website URL
- [ ] Check health endpoint
- [ ] Test a prediction
- [ ] Verify model accuracy shows 97.32%
- [ ] Check ZAMCELCO branding appears
- [ ] Test with different rooms
- [ ] Verify cost calculations
- [ ] Check prediction history works

---

## 🎉 You're Ready!

Run these commands to deploy:

```bash
git add .
git commit -m "Update to ZAMCELCO realistic data (March-April 2024) - 97.32% accuracy"
git push origin main
```

Then wait 2-3 minutes and your website will be updated! 🚀

---

**Last Updated:** April 29, 2026  
**Deployment:** Render  
**Status:** Ready to Deploy ✅

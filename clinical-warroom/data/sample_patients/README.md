# Sample Patient Files - Treatment Urgency Testing

This directory contains diverse patient samples designed to trigger different treatment urgency recommendations from the multi-agent debate system.

## 📋 Patient Samples Overview

### 🚨 CRITICAL - IMMEDIATE TREATMENT

**File:** `critical_emergency_patient.txt`  
**Patient:** Michael Chen, 67M  
**Condition:** Acute STEMI (ST-Elevation Myocardial Infarction)  
**Expected Urgency:** 🚨 **IMMEDIATE TREATMENT NEEDED**

**Key Indicators:**
- Active chest pain (9/10)
- ECG showing ST elevations
- Multiple risk factors (diabetes, smoking, hypertension)
- Time-critical emergency (<90 min door-to-balloon)
- Requires emergency cardiac catheterization

---

### ⚠️ MODERATE - ESCALATE TO SPECIALIST

**File:** `moderate_risk_patient.txt`  
**Patient:** Robert Martinez, 58M  
**Condition:** Controlled hypertension with prediabetes  
**Expected Urgency:** ⚠️ **ESCALATE TO SPECIALIST**

**Key Indicators:**
- Prediabetic (HbA1c 5.8%)
- Family history of heart disease
- Low HDL cholesterol
- Recent fatigue
- Requires specialist evaluation and lifestyle intervention

---

### 📋 LOW RISK - CONTINUE MONITORING

**File 1:** `healthy_patient.txt`  
**Patient:** Sarah Johnson, 32F  
**Condition:** Healthy, routine wellness checkup  
**Expected Urgency:** 📋 **CONTINUE MONITORING**

**Key Indicators:**
- All vitals normal
- No chronic conditions
- Regular exercise, healthy lifestyle
- Excellent lab values
- Routine preventive care only

**File 2:** `excellent_health_patient.txt`  
**Patient:** Emily Thompson, 45F  
**Condition:** Exceptionally healthy, model patient  
**Expected Urgency:** 📋 **CONTINUE MONITORING**

**Key Indicators:**
- Optimal cardiovascular markers
- Very active lifestyle (yoga 4x/week)
- No family history of early disease
- All screenings normal
- No interventions needed

---

## 🧪 Testing Instructions

### How to Test Different Urgency Levels:

1. **Upload Patient:**
   - Go to "Patient Registration" tab
   - Upload one of the `.txt` files above

2. **Query Patient:**
   - Switch to "Clinical Query" tab
   - Select the uploaded patient
   - Ask: "Does this patient need immediate treatment?"

3. **Observe Debate:**
   - Watch the 3 agents argue (Proponent, Skeptic, Mediator)
   - See 3 rounds of debate
   - Final consensus card shows urgency level

### Expected Results:

| Patient | Condition | Proponent Stance | Skeptic Stance | Final Urgency |
|---------|-----------|------------------|----------------|---------------|
| Michael Chen | STEMI | Push for immediate cath lab | Note time sensitivity | 🚨 IMMEDIATE |
| Robert Martinez | Prediabetes + HTN | Suggest specialist | Question urgency | ⚠️ ESCALATE |
| Sarah Johnson | Healthy | Minimal concern | Advise routine care | 📋 MONITOR |
| Emily Thompson | Excellent health | No intervention needed | Agree on monitoring | 📋 MONITOR |

---

## 💡 Use Cases

- **Demonstrations:** Show how AI handles different severity levels
- **Testing:** Verify debate logic works correctly
- **Training:** Understand multi-agent decision-making
- **Validation:** Confirm system doesn't over/under-treat

---

## 📝 Notes

- Patients are fictional but medically realistic
- Clinical details are accurate for their conditions
- Urgency levels match evidence-based guidelines
- Use these to validate the debate system's reasoning

Created for Clinical Decision Support Platform v2.0

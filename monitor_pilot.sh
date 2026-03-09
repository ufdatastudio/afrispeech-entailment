#!/bin/bash
# Monitor pilot test job progress

JOB_ID=23445440
LOG_DIR="/orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs/Kimi/interview_nli_pilot/logs"
RESULTS_DIR="/orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs/Kimi/interview_nli_pilot/results"

echo "======================================================================"
echo "🔍 Monitoring Pilot Test: Interview NLI"
echo "======================================================================"
echo "Job ID: $JOB_ID"
echo ""

# Check job status
echo "📊 Job Status:"
squeue -j $JOB_ID 2>/dev/null || echo "   Job completed or not found"
echo ""

# Show recent log output
echo "📝 Recent Log Output:"
echo "----------------------------------------------------------------------"
if [ -f "${LOG_DIR}/infer_${JOB_ID}.out" ]; then
    tail -30 "${LOG_DIR}/infer_${JOB_ID}.out"
else
    echo "   Log file not yet created: ${LOG_DIR}/infer_${JOB_ID}.out"
fi
echo ""

# Check for results
echo "📁 Results:"
if [ -f "${RESULTS_DIR}/Kimi_interview_nli_pilot.jsonl" ]; then
    RESULT_COUNT=$(wc -l < "${RESULTS_DIR}/Kimi_interview_nli_pilot.jsonl")
    echo "   ✅ Results file exists: ${RESULTS_DIR}/Kimi_interview_nli_pilot.jsonl"
    echo "   📊 Predictions: $RESULT_COUNT / 27 expected"
else
    echo "   ⏳ Results file not yet created"
fi
echo ""

echo "======================================================================"
echo "Commands:"
echo "   Watch log:   tail -f ${LOG_DIR}/infer_${JOB_ID}.out"
echo "   Check queue: squeue -u c.okocha"
echo "   View errors: cat ${LOG_DIR}/infer_${JOB_ID}.err"
echo "======================================================================"

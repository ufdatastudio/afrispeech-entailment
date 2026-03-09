#!/bin/bash
# Wait for pilot job to complete and show results

echo "Waiting for job 23447874 to complete..."
echo ""

# Wait for results
MAX_WAIT=1200  # 20 minutes
ELAPSED=0
INTERVAL=10

while [ $ELAPSED -lt $MAX_WAIT ]; do
    # Check if results file exists
    if [ -f "/orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs/Kimi/interview_nli_pilot/results/Kimi_interview_nli_pilot.jsonl" ]; then
        echo "✅ Results file created!"
        break
    fi
    
    # Check job status
    STATUS=$(squeue -j 23447874 2>/dev/null | tail -1)
    if [ -z "$STATUS" ]; then
        echo "Job completed (no longer in queue)"
        break
    fi
    
    echo -ne "\r⏳ Waiting... ($ELAPSED seconds elapsed)"
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

echo ""
echo ""
echo "======================================================================"
echo "📊 RESULTS"
echo "======================================================================"
echo ""

# Check results
RESULTS_FILE="/orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs/Kimi/interview_nli_pilot/results/Kimi_interview_nli_pilot.jsonl"

if [ -f "$RESULTS_FILE" ]; then
    COUNT=$(wc -l < "$RESULTS_FILE")
    echo "✅ Results file found: $RESULTS_FILE"
    echo "   Predictions: $COUNT / 27"
    echo ""
    
    # Show first few lines
    echo "📝 First 3 predictions:"
    head -3 "$RESULTS_FILE" | python3 -m json.tool 2>/dev/null | head -30
else
    echo "❌ Results file not found"
    echo ""
    echo "📋 Check logs:"
    echo "   tail -100 /orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs/Kimi/interview_nli_pilot/logs/infer_*.out"
fi

echo ""
echo "======================================================================"

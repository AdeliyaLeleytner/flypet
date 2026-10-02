"""Question wording for instruction generalization; no state-to-text inference rules."""

SUMMARY = "Briefly describe your current neural response, without exact numbers or spike counts."
CHANGE_QUALITATIVE = (
    "Briefly compare the current neural response with the reference, without numbers."
)
AUDIT = "Report the current approach_hz, avoid_hz and valence as JSON. Valence is (approach_hz-avoid_hz)/(approach_hz+avoid_hz+1), positive when approach exceeds avoidance. Use one decimal for rates and three for valence."
CHANGE = "How did the response change relative to the reference observation?"

QUESTIONS = {
    "summary": [
        "Describe your current neural response in a short sentence.",
        SUMMARY,
        "How are your approach and avoidance outputs behaving right now? Give a short qualitative answer.",
        "In plain language, how does the current neural response lean?",
        "Describe the current response without exact numbers.",
        "Give one short qualitative sentence about your current neural response. Do not include any numbers or spike counts.",
        "Tell me which way your neural response leans, keeping the answer brief.",
        "What is the overall direction of the current neural response?",
    ],
    "audit": [
        AUDIT,
        "Give the current neural readout as JSON with approach_hz, avoid_hz and valence. Rates use one decimal and valence three decimals.",
        "Read the current observation and return only a JSON object containing approach_hz, avoid_hz, and valence.",
        "What are the current approach and avoidance firing rates and their normalized balance? Use JSON keys approach_hz, avoid_hz, valence.",
        "Provide the current MBON readout: approach_hz, avoid_hz, valence. Return one JSON object.",
        "Inspect the current neural output and report its approach_hz, avoid_hz and valence in JSON.",
    ],
    "change": [
        CHANGE,
        "By how much did the neural balance move relative to the reference?",
        "Report the signed numerical change in neural balance.",
        "Give the difference between the current and reference neural balance.",
        "How has the neural balance moved? State its signed change and direction.",
        "Compare the observations using a signed numerical change estimate.",
    ],
    "change_qualitative": [
        CHANGE_QUALITATIVE,
        "Has the balance shifted toward approach or avoidance relative to the saved observation?",
        "What changed since the reference? Use a short qualitative description.",
        "Tell me the direction of the change, without quoting counts or numerical values.",
        "Describe only the direction of the neural change.",
        "How does your present response differ from the saved observation? Give a brief qualitative answer.",
    ],
}

# Not sampled during training; fixed before the fresh panel is opened.
HELD_SUMMARY = "Describe which side dominates the present neural output: approach, avoidance, or neither. Keep it qualitative."
HELD_CHANGE_QUALITATIVE = "How has the approach-versus-avoidance balance shifted since the reference? State only the direction."

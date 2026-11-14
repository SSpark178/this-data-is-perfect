def inspect_batch(state: BatchState) -> dict:
    """Measure the lot and decide whether it is ready to leave the floor."""
    ready = MOISTURE_MIN <= state["moisture"] <= MOISTURE_MAX
    return {"inspections": state["inspections"] + 1, "approved": ready}

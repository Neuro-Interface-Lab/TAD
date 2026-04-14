"""
Example 7: Working with Triggers and TimeSlots
===============================================

This example shows how to create, manipulate, and save trigger events
(e.g., stimulation times, behavioral events).
"""

from tad import Triggers, TimeSlot, load_triggers_from_json

# Create an empty trigger object
triggers = Triggers(slots=[])

# Add triggers using different approaches

# Method 1: Add a trigger using tstart and duration
triggers.add_timed_slot(
    tstart=1.0,
    duration=0.5,
    ID=\"stim_1\",
    description=\"First stimulation\",
)

# Method 2: Add a trigger using start and end times
triggers.add_interval_slot(
    start=3.0,
    end=3.2,
    ID=\"stim_2\",
    description=\"Second stimulation\",
)

# Method 3: Add a trigger manually via TimeSlot
triggers.slots.append(
    TimeSlot(
        start=5.0,
        end=5.1,
        ID=\"stim_3\",
        description=\"Third stimulation\",
    )
)

# Display all triggers
print(f\"Total triggers: {len(triggers.slots)}\")
for i, slot in enumerate(triggers.slots):
    print(f\"  Trigger {i}: {slot.start:.2f}s - {slot.end:.2f}s, ID={slot.ID}\")

# Save triggers to JSON
triggers.save2json(\"my_triggers.json\")
print(\"\\nSaved triggers to my_triggers.json\")

# Load triggers back from JSON
loaded = load_triggers_from_json(\"my_triggers.json\")
print(f\"Loaded {len(loaded.slots)} triggers from file\")
assert len(loaded.slots) == len(triggers.slots), \"Trigger count mismatch!\"

# Query triggers in a time window
window_start, window_end = 0.5, 4.0
triggers_in_window = [
    s for s in triggers.slots
    if s.start < window_end and s.end > window_start
]
print(f\"\\nTriggers in window [{window_start}, {window_end}]: {len(triggers_in_window)}\")

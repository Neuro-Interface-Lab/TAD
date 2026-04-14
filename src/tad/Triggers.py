from dataclasses import dataclass
from typing import List, Optional

import json


def load_triggers_from_json(filepath: str) -> 'Triggers':
    """
    Load triggers from a JSON file.

    Parameters
    ----------
    filepath
        Path to the JSON file from which the triggers will be loaded.

    Returns
    -------
    TimeSlot
        A list of TimeSlot objects representing the triggers.
    """
    try:
        loaded_triggers = Triggers(slots=[])
        loaded_triggers.load_from_json(filepath)
        return loaded_triggers
    except Exception as e:
        print(f"Error loading triggers from JSON: {e}")
        return Triggers(slots=[])


@dataclass
class TimeSlot:
    """
    Represents a time slot with a start and end time.

    Parameters
    ----------
    start
        Start time of the slot.
    end
        End time of the slot.
    ID 
        Optional identifier for the slot
    description
        Optional description for the slot
    blank
        Optional flag to indicate if the slot should be blanked out in the analysis (e.g., for artifact removal)
    """
    start: float
    end: float
    ID: Optional[int] = None 
    description: Optional[str] = None 
    blank: Optional[bool] = False 

    def duration(self) -> float:
        """
        Compute the duration of the time slot.

        Returns
        -------
        float
            Duration of the slot (end - start).
        """
        return self.end - self.start
    
    def __eq__(self, other_timeslot) -> bool:
        if self.start == other_timeslot.start and self.end == other_timeslot.end:
            return True
        return False
    
    def __str__(self) -> str:
        return f"TimeSlot(start={self.start}, end={self.end}, ID={self.ID}, description={self.description}, blank = {self.blank})"
    
    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class Triggers:
    """
    Container for a list of time slots representing triggers.

    Parameters
    ----------
    slots
        List of TimeSlot objects representing the triggers.
    """
    slots: List[TimeSlot]

    def add_timed_slot(self, tstart:float, duration:float, ID: Optional[int] = None, description: Optional[str] = None, blank: Optional[bool] = False) -> None:
        """
        Add a new time slot to the triggers.

        Parameters
        ----------
        tstart
            Start time of the new slot.
        duration
            Duratino of the new slot.
        ID
            Optional identifier for the new slot.
        description
            Optional description for the new slot.
        blank
            Optional flag to indicate if the slot should be blanked out in the analysis (e.g., for artifact removal).
        """
        new_slot = TimeSlot(start=tstart, end=tstart+duration, ID=ID, description=description, blank = blank)
        self.slots.append(new_slot)

    def add_interval_slot(self, start:float, end:float, ID: Optional[int] = None, description: Optional[str] = None, blank: Optional[bool] = False) -> None:
        """
        Add a new time slot to the triggers.

        Parameters
        ----------
        start
            Start time of the new slot.
        end
            End time of the new slot.
        ID
            Optional identifier for the new slot.
        description
            Optional description for the new slot.
        blank
            Optional flag to indicate if the slot should be blanked out in the analysis (e.g., for artifact removal).
        """
        new_slot = TimeSlot(start=start, end=end, ID=ID, description=description, blank = blank)
        self.slots.append(new_slot)

    def __str__(self) -> str:
        return f"Triggers(slots={self.slots})"
    
    def __repr__(self) -> str:
        return self.__str__()

    def sort_slots(self) -> None:
        """
        Sort the time slots in ascending order based on their start times.
        """
        self.slots.sort(key=lambda slot: slot.start)

    def save2json(self, filepath: str) -> None:
        """
        Save the triggers to a JSON file.

        Parameters
        ----------
        filepath
            Path to the JSON file where the triggers will be saved.
        """
        with open(filepath, 'w') as f:
            json.dump([slot.__dict__ for slot in self.slots], f, indent=4)
        
    def load_from_json(self, filepath: str) -> None:
        """
        Load triggers from a JSON file.

        Parameters
        ----------
        filepath
            Path to the JSON file from which the triggers will be loaded.
        """
        with open(filepath, 'r') as f:
            slots_data = json.load(f)
            self.slots = [TimeSlot(**slot) for slot in slots_data]

    def check_overlaps(self) -> List[tuple]:
        """
        Check for overlapping time slots.

        Returns
        -------
        List[tuple]
            A list of tuples, where each tuple contains the indices of the overlapping slots.
        """
        overlaps = []
        for i in range(len(self.slots)):
            for j in range(i + 1, len(self.slots)):
                if self.slots[i].end > self.slots[j].start and self.slots[i].start < self.slots[j].end:
                    overlaps.append((i, j))
        return overlaps

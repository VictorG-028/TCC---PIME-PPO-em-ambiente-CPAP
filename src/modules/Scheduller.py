from typing import List
from numpy import cumsum as np_cumsum, searchsorted as np_searchsorted
from numpy.typing import NDArray

class Scheduller:

    def __init__(
            self,
            set_points: List[float] = [],
            intervals: List[int] = [], # interval in steps
            ) -> None:
        assert len(set_points) == len(intervals), "set_points and intervals must have save lenght"

        # Example:
        # intervals = [500, 100, 200]
        # cumulative_intervals = [500, 600, 800]

        self.set_points = set_points
        self.intervals = intervals
        self.cumulative_intervals: NDArray = np_cumsum(self.intervals) 
        self.intervals_sum = sum(self.intervals)
        
    def get_set_point_at(self, *, step: int) -> float:
        """
        Returns the set point value for the given step.

        Args:
            step (int): Current step.

        Returns:
            float: The corresponding set point value.

        Raises:
            AssertionError: If step is negative.
        """

        assert step >= 0, "Step must not be negative"

        
        max_step = self.cumulative_intervals[-1]
        if step >= max_step:
            return self.set_points[-1]
        
        for i, cum_interval in enumerate(self.cumulative_intervals): 
            if step < cum_interval: 
                return self.set_points[i]
        
        # More efficient version for looong self.cumulative_intervals np array
        # index = np_searchsorted(self.cumulative_intervals, step, side='right')
        # return self.set_points[index]

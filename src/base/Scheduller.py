from typing import List
from numpy import cumsum as np_cumsum
from numpy.typing import NDArray

class Scheduller:

    def __init__(
            self,
            set_points: List[float] = [],
            intervals: List[int] = [], # interval in steps
            ) -> None:
        assert len(set_points) == len(intervals), "set_points and intervals must have save lenght"

        self.set_points = set_points
        self.intervals = intervals
        self.cumulative_intervals: NDArray = np_cumsum(self.intervals)
        self.intervals_sum = sum(self.intervals)
        
    def get_set_point_at(self, *, step: int) -> float:
        assert step >= 0, "Step must not be negative"

        cumulative_intervals = np_cumsum(self.intervals)

        max_step = cumulative_intervals[-1]
        if step >= max_step:
            return self.set_points[-1]
        
        for i, cum_interval in enumerate(cumulative_intervals): # [500, 100, 200]
            if step < cum_interval: 
                return self.set_points[i]

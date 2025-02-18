from typing import List, Literal
from numpy import cumsum as np_cumsum, searchsorted as np_searchsorted
from numpy.typing import NDArray

class Scheduller:

    def __init__(
            self,
            set_points: List[float] = [],
            intervals: List[int] = [], # interval in steps
            mode: Literal["loop set points", "repeat last"] = "loop set points",
            ) -> None:
        assert len(set_points) == len(intervals), "set_points and intervals must have same length"

        # Example:
        # intervals = [500, 100, 200]
        # cumulative_intervals = [500, 600, 800]
        # intervals_sum = 800

        self.set_points = set_points
        self.intervals = intervals
        self.cumulative_intervals: NDArray = np_cumsum(self.intervals)
        self.intervals_sum = self.cumulative_intervals[-1] # sum(self.intervals)

        if mode == "repeat last":
            self.mode_fn = lambda t: self.set_points[-1]
        elif mode == "loop set points":
            self.mode_fn = self._loop_set_points
        else:
            raise ValueError("Invalid mode. Must choose between 'loop set points' and 'repeat last'")
        
    def _loop_set_points(self, step: int) -> float:
        step_mod = step % self.intervals_sum

        # for i, cum_interval in enumerate(self.cumulative_intervals):
        #     if step_mod <= cum_interval:
        #         return self.set_points[i]

        # More efficient version of above code for looong self.cumulative_intervals np array
        index = np_searchsorted(self.cumulative_intervals, step_mod, side='right')
        return self.set_points[index]
        

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

        if step >= self.intervals_sum:
            return self.mode_fn(step)
        
        # for i, cum_interval in enumerate(self.cumulative_intervals): 
        #     if step < cum_interval: 
        #         return self.set_points[i]
        
        # More efficient version of above code for looong self.cumulative_intervals np array
        index = np_searchsorted(self.cumulative_intervals, step, side='right')
        return self.set_points[index]


    def _test_loop_set_points_mode_fn() -> None:

        s = Scheduller(
            intervals=[5, 1, 4],
            set_points=[5, 15, 10],
            mode="loop set points"
        )

        # s.intervals_sum = 10 = 5 + 1 + 4
        results = []
        how_many_times_to_loop = 2
        correct_resutls = [5, 5, 5, 5, 5, 15, 10, 10, 10, 10] * how_many_times_to_loop

        for t in range(0 * s.intervals_sum, 1 * s.intervals_sum):
            set_point = s.get_set_point_at(step=t)
            results.append(set_point)
        #     print(sp)
        # print("-------")

        for t in range(1 * s.intervals_sum, 2 * s.intervals_sum):
            set_point = s.get_set_point_at(step=t)
            results.append(set_point)
        #     print(sp)
        # print("-------")
        
        assert results == correct_resutls, "Loop set points mode test failed"
        print("Loop set points mode test passed")


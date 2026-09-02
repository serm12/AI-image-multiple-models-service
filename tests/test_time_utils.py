import tempfile
import unittest

from app.utils.task_utils import generate_task_dir
from app.utils.time_utils import now_china


class ChinaTimeTests(unittest.TestCase):
    def test_now_china_has_utc_plus_eight_offset(self):
        current = now_china()
        self.assertEqual(current.utcoffset().total_seconds(), 8 * 60 * 60)
        self.assertEqual(current.tzname(), "CST")

    def test_task_timestamp_uses_china_time(self):
        with tempfile.TemporaryDirectory() as tasks_dir:
            before = now_china().strftime("%Y%m%d_%H%M")
            task_id, _task_dir, timestamp = generate_task_dir(tasks_dir)
            after = now_china().strftime("%Y%m%d_%H%M")

        self.assertIn(timestamp[:13], {before, after})
        self.assertTrue(task_id.startswith(timestamp))


if __name__ == "__main__":
    unittest.main()

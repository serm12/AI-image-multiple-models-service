from datetime import datetime
import tempfile
import unittest

from app.utils.task_utils import generate_task_dir
from app.utils.time_utils import now_china, now_utc


class ChinaTimeTests(unittest.TestCase):
    def test_now_china_has_utc_plus_eight_offset(self):
        current = now_china()
        self.assertEqual(current.utcoffset().total_seconds(), 8 * 60 * 60)
        self.assertEqual(current.tzname(), "CST")

    def test_task_timestamp_uses_utc(self):
        with tempfile.TemporaryDirectory() as tasks_dir:
            before = now_utc()
            task_id, _task_dir, timestamp = generate_task_dir(tasks_dir)
            after = now_utc()

        created = datetime.fromisoformat(timestamp)
        self.assertLessEqual(before, created)
        self.assertLessEqual(created, after)
        self.assertEqual(created.utcoffset().total_seconds(), 0)
        self.assertTrue(task_id.startswith(created.strftime("%Y%m%d_%H%M%S")))


if __name__ == "__main__":
    unittest.main()

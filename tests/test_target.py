import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from target import Person, Target, csv_row, validate_box
from track import (Detector, athlete_boxes, cached_selection, choose_device, clear_white_candidate,
                   load_selections, log_transition, prompt_selection, update_selection_cache)


class TargetTests(unittest.TestCase):
    def setUp(self):
        self.white = Person(7, (10, 10, 50, 110), 0.9, 0.8, 0.0, (1.0, 0.0))
        self.blue = Person(9, (150, 10, 190, 110), 0.9, 0.1)
        self.target = Target()
        self.target.select(self.white.box, [self.white, self.blue], [self.white.box, self.blue.box])

    def test_tracking_continuity_and_csv(self):
        moved = Person(7, (12, 11, 52, 111), 0.9, 0.75)
        box = self.target.update([moved, self.blue], [moved.box, self.blue.box])
        self.assertEqual(box, moved.box)
        self.assertEqual(csv_row(1.5, self.target, box), ['1.500000', 1, *box, 'tracking', 7])

    def test_lost_never_adopts_nearby_white_person_or_returned_id(self):
        replacement = Person(20, self.white.box, 0.9, 0.8)
        for _ in range(45):
            self.assertIsNone(self.target.update([replacement], [replacement.box]))
        self.assertEqual(self.target.status, 'lost')
        self.assertEqual(self.target.track_id, 7)
        self.assertIsNone(self.target.update([self.white], [self.white.box]))
        buffer = io.StringIO()
        csv.writer(buffer).writerow(csv_row(2, self.target, None))
        self.assertEqual(buffer.getvalue().strip(), '2.000000,1,,,,,lost,')

    def test_overlap_with_an_untracked_person_recovers_same_id_after_clear_frames(self):
        other_box = (25, 10, 65, 110)
        self.assertIsNone(self.target.update([self.white], [self.white.box, other_box]))
        self.assertEqual(self.target.status, 'uncertain')
        self.assertIsNone(self.target.update([self.white], [self.white.box]))
        self.assertIsNone(self.target.update([self.white], [self.white.box]))
        self.assertEqual(self.target.update([self.white], [self.white.box]), self.white.box)
        self.assertEqual(self.target.status, 'tracking')

    def test_overlap_reidentifies_new_id_only_after_three_clear_frames(self):
        other_box = (25, 10, 65, 110)
        self.target.update([self.white], [self.white.box, other_box])
        replacement = Person(20, self.white.box, 0.9, 0.8, 0.0, (1.0, 0.0))
        self.assertIsNone(self.target.update([replacement], [replacement.box]))
        self.assertEqual((self.target.status, self.target.track_id), ('uncertain', 7))
        self.assertIsNone(self.target.update([replacement], [replacement.box]))
        self.assertEqual(self.target.update([replacement], [replacement.box]), replacement.box)
        self.assertEqual((self.target.status, self.target.track_id), ('tracking', 20))
        self.assertEqual(self.target.target_id, 1)

    def test_overlap_does_not_reidentify_blue_or_two_white_candidates(self):
        self.target.update([self.white], [self.white.box, (25, 10, 65, 110)])
        blue = Person(20, self.white.box, 0.9, 0.1, 0.6)
        for _ in range(3):
            self.assertIsNone(self.target.update([blue], [blue.box]))
        self.assertEqual(self.target.track_id, 7)
        another_white = Person(21, (13, 10, 53, 110), 0.9, 0.8, 0.0, (1.0, 0.0))
        for _ in range(3):
            self.assertIsNone(self.target.update([Person(20, self.white.box, 0.9, 0.8, 0.0,
                                                     (1.0, 0.0)), another_white],
                                                  [self.white.box, another_white.box]))
        self.assertEqual(self.target.track_id, 7)

    def test_new_id_requires_visual_match_and_motion(self):
        self.target.update([self.white], [self.white.box, (25, 10, 65, 110)])
        wrong_texture = Person(20, self.white.box, 0.9, 0.8, 0.0, (0.0, 1.0))
        for _ in range(3):
            self.assertIsNone(self.target.update([wrong_texture], [wrong_texture.box]))
        self.assertEqual(self.target.track_id, 7)
        far = Person(21, (100, 10, 140, 110), 0.9, 0.8, 0.0, (1.0, 0.0))
        for _ in range(3):
            self.assertIsNone(self.target.update([far], [far.box]))
        self.assertEqual(self.target.track_id, 7)

    def test_new_id_must_follow_predicted_motion_not_just_be_nearby(self):
        self.target.velocity = (15.0, 0.0)
        self.target.update([self.white], [self.white.box, (25, 10, 65, 110)])
        for _ in range(8):
            self.target.update([], [])
        stationary = Person(20, self.white.box, 0.9, 0.8, 0.0, (1.0, 0.0))
        for _ in range(3):
            self.assertIsNone(self.target.update([stationary], [stationary.box]))
        self.assertEqual(self.target.track_id, 7)

    def test_track_change_and_recovery_events_keep_target_id(self):
        self.target.update([self.white], [self.white.box, (25, 10, 65, 110)])
        new = Person(20, self.white.box, 0.9, 0.8, 0.0, (1.0, 0.0))
        self.target.update([new], [new.box])
        self.target.update([new], [new.box])
        before = (self.target.status, self.target.track_id, self.target.reason)
        self.target.update([new], [new.box])
        buffer = io.StringIO()
        log_transition(csv.writer(buffer), 30, 30, self.target, before)
        rows = list(csv.reader(io.StringIO(buffer.getvalue())))
        self.assertEqual([row[5] for row in rows], ['track_id_changed', 'recovered'])
        self.assertTrue(all(row[2:5] == ['1', '7', '20'] for row in rows))
        self.assertTrue(all(row[1] == '1.000000' for row in rows))

    def test_lost_event_has_timestamp_and_stable_target_id(self):
        for _ in range(44):
            self.target.update([], [])
        before = (self.target.status, self.target.track_id, self.target.reason)
        self.target.update([], [])
        buffer = io.StringIO()
        log_transition(csv.writer(buffer), 45, 30, self.target, before)
        row = next(csv.reader(io.StringIO(buffer.getvalue())))
        self.assertEqual(row[:6], ['45', '1.500000', '1', '7', '7', 'lost'])

    def test_camera_cut_during_overlap_requires_user_selection(self):
        self.target.update([self.white], [self.white.box, (25, 10, 65, 110)])
        self.assertIsNone(self.target.update([self.white], [self.white.box], camera_cut=True))
        self.assertEqual(self.target.reason, 'Possible camera cut')
        self.assertIsNone(self.target.update([self.white], [self.white.box]))

    def test_small_box_touch_does_not_stop_tracking(self):
        self.assertEqual(self.target.update([self.white], [self.white.box, (40, 10, 80, 110)]), self.white.box)

    def test_cut_latches_uncertainty_even_with_same_id(self):
        self.assertIsNone(self.target.update([self.white], [self.white.box], camera_cut=True))
        self.assertEqual(self.target.reason, 'Possible camera cut')
        self.assertIsNone(self.target.update([self.white], [self.white.box]))

    def test_jump_size_color_and_confidence_are_rejected(self):
        for person in (
            Person(7, (250, 10, 290, 110), 0.9, 0.8),
            Person(7, (10, 10, 100, 210), 0.9, 0.8),
            Person(7, self.white.box, 0.9, 0.1),
            Person(7, self.white.box, 0.9, 0.8, 0.5),
            Person(7, self.white.box, 0.2, 0.8),
        ):
            with self.subTest(person=person):
                self.setUp()
                self.assertIsNone(self.target.update([person], [person.box]))
                self.assertEqual(self.target.status, 'uncertain')

    def test_overlap_allows_partial_box_then_recovers_only_original_id(self):
        self.target.update([self.white], [self.white.box, (25, 10, 65, 110)])
        partial = Person(7, (20, 20, 40, 80), 0.8, 0.7, 0.02)
        self.assertIsNone(self.target.update([partial], [partial.box, (23, 20, 45, 80)]))
        complete = Person(7, (15, 12, 55, 112), 0.9, 0.75, 0.03)
        self.assertIsNone(self.target.update([complete], [complete.box]))
        self.assertIsNone(self.target.update([complete], [complete.box]))
        self.assertEqual(self.target.update([complete], [complete.box]), complete.box)

    def test_overlap_rejects_blue_appearance_even_with_same_id(self):
        self.target.update([self.white], [self.white.box, (25, 10, 65, 110)])
        blue = Person(7, self.white.box, 0.9, 0.65, 0.5)
        for _ in range(6):
            self.assertIsNone(self.target.update([blue], [blue.box]))
        self.assertEqual(self.target.reason, 'Jersey appearance changed')

    def test_temporary_merged_jersey_recovers_after_three_clear_frames(self):
        self.target.update([self.white], [self.white.box, (25, 10, 65, 110)])
        merged = Person(7, self.white.box, 0.9, 0.4, 0.3)
        for _ in range(3):
            self.assertIsNone(self.target.update([merged], [merged.box]))
        self.assertEqual(self.target.reason, 'People overlap')
        for _ in range(2):
            self.assertIsNone(self.target.update([self.white], [self.white.box]))
        self.assertEqual(self.target.update([self.white], [self.white.box]), self.white.box)

    def test_blue_sleeve_during_overlap_does_not_permanently_stop_target(self):
        contaminated = Person(7, self.white.box, 0.9, 0.4, 0.5)
        boxes = [self.white.box, (25, 10, 65, 110)]
        self.assertIsNone(self.target.update([contaminated], boxes))
        self.assertEqual(self.target.reason, 'People overlap')
        for _ in range(2):
            self.assertIsNone(self.target.update([self.white], [self.white.box]))
        self.assertEqual(self.target.update([self.white], [self.white.box]), self.white.box)

    def test_only_explicit_reselection_changes_id(self):
        self.target.update([], [])
        replacement = Person(20, self.white.box, 0.9, 0.8)
        self.target.select(replacement.box, [replacement], [replacement.box])
        self.assertEqual(self.target.track_id, 20)
        self.assertEqual(self.target.target_id, 1)
        self.assertEqual(self.target.status, 'tracking')

    def test_reselection_by_visible_id_requires_manual_input(self):
        self.target.update([], [])
        replacement = Person(20, self.white.box, 0.9, 0.8)
        with patch('track.snapshot'), patch('sys.stdin.isatty', return_value=True), \
                patch('builtins.input', return_value='id 20'):
            prompt_selection(None, [replacement], [replacement.box], None, 20, self.target)
        self.assertEqual((self.target.status, self.target.track_id), ('tracking', 20))

    def test_selection_must_match_a_clear_person(self):
        for people, boxes, box in (
            ([], [], self.white.box),
            ([self.white], [self.white.box], (300, 10, 340, 110)),
            ([self.white, Person(9, (20, 10, 60, 110))],
             [self.white.box, (20, 10, 60, 110)], self.white.box),
            ([Person(7, self.white.box, 0.2)], [self.white.box], self.white.box),
        ):
            with self.subTest(box=box):
                with self.assertRaises(ValueError):
                    self.target.select(box, people, boxes)

    def test_referee_overlap_does_not_block_white_athlete_selection(self):
        referee = (35, 10, 65, 60)
        detections = SimpleNamespace(xyxy=[self.white.box, referee], confidence=[0.9, 0.8])
        with patch('track.jersey_fractions', side_effect=[(0.8, 0.0), (0.02, 0.01)]):
            boxes = athlete_boxes(None, detections)
        self.assertEqual(boxes, [self.white.box])
        self.target.select(self.white.box, [self.white], boxes)
        self.assertEqual(self.target.status, 'tracking')

    def test_prompt_waits_for_a_clear_white_person(self):
        overlap = (25, 10, 65, 110)
        self.assertFalse(clear_white_candidate([self.white], [self.white.box, overlap]))
        self.assertFalse(clear_white_candidate([self.blue], [self.blue.box]))
        self.assertTrue(clear_white_candidate([self.white], [self.white.box]))

    def test_invalid_coordinates(self):
        for box in ((1, 2, 3), (-1, 0, 10, 10), (0, 0, 0, 10),
                    (0, 0, float('nan'), 10), (0, 0, 101, 10)):
            with self.subTest(box=box), self.assertRaises(ValueError):
                validate_box(box, 100, 100)

    def test_unselected_coordinates_and_id_are_empty(self):
        self.assertEqual(csv_row(0, Target(), None), ['0.000000', '', '', '', '', '', 'unselected', ''])


class DeviceTests(unittest.TestCase):
    def fake_torch(self, available):
        return SimpleNamespace(__version__='fake', version=SimpleNamespace(cuda=None),
                               device=lambda name: name,
                               cuda=SimpleNamespace(is_available=lambda: available))

    def test_auto_uses_cpu_without_cuda(self):
        with patch.dict('sys.modules', torch=self.fake_torch(False)), patch('builtins.print'):
            self.assertEqual(choose_device('auto'), 'cpu')

    def test_explicit_cuda_is_not_silently_ignored(self):
        with patch.dict('sys.modules', torch=self.fake_torch(False)), patch('builtins.print'):
            with self.assertRaises(ValueError):
                choose_device('cuda')

    def test_auto_falls_back_if_cuda_operation_fails(self):
        torch = self.fake_torch(True)
        def failed_operation(*args, **kwargs):
            raise RuntimeError('CUDA driver failed')
        torch.ones = failed_operation
        with patch.dict('sys.modules', torch=torch), patch('builtins.print'):
            self.assertEqual(choose_device('auto'), 'cpu')


class DetectorTests(unittest.TestCase):
    def test_yolov8_person_boxes_feed_existing_tracker_format(self):
        import numpy as np
        import torch
        boxes = MagicMock()
        boxes.__len__.return_value = 1
        boxes.xyxy = torch.tensor([[10., 20., 50., 100.]])
        boxes.conf = torch.tensor([0.2])
        boxes.cls = torch.tensor([0.])
        model = MagicMock()
        model.predict.return_value = [SimpleNamespace(boxes=boxes)]
        with tempfile.TemporaryDirectory() as folder:
            weights = Path(folder) / 'yolov8n.pt'
            weights.touch()
            with patch.dict('sys.modules', ultralytics=SimpleNamespace(YOLO=lambda *a, **k: model)):
                detector = Detector(weights, 'cpu')
            detections = detector.detect(np.zeros((120, 80, 3), dtype=np.uint8))
        self.assertEqual(detections.xyxy.tolist(), [[10., 20., 50., 100.]])
        self.assertEqual(detections.class_id.tolist(), [0])
        self.assertAlmostEqual(float(detections.confidence[0]), 0.2, places=6)
        self.assertEqual(model.predict.call_args.kwargs['classes'], [0])
        self.assertEqual(model.predict.call_args.kwargs['conf'], 0.1)

    def test_yolov8_empty_frame_returns_empty_detections(self):
        model = MagicMock()
        model.predict.return_value = [SimpleNamespace(boxes=[])]
        with tempfile.TemporaryDirectory() as folder:
            weights = Path(folder) / 'yolov8n.pt'
            weights.touch()
            with patch.dict('sys.modules', ultralytics=SimpleNamespace(YOLO=lambda *a, **k: model)):
                detector = Detector(weights, 'cpu')
            self.assertEqual(len(detector.detect(None)), 0)

class SelectionTests(unittest.TestCase):
    def test_first_selection_survives_new_run_and_can_be_reset(self):
        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder) / 'match.mp4'
            video.write_bytes(b'first video')
            cache = Path(folder) / 'cache.json'
            self.assertIsNone(cached_selection(video, cache))
            update_selection_cache(video, 300, (10, 10, 50, 110), cache)
            self.assertEqual(cached_selection(video, cache), (300, (10., 10., 50., 110.)))
            video.write_bytes(b'different video')
            self.assertIsNone(cached_selection(video, cache))
            update_selection_cache(video, 400, (20, 10, 60, 110), cache)
            self.assertEqual(cached_selection(video, cache)[0], 400)
            update_selection_cache(video, path=cache)
            self.assertIsNone(cached_selection(video, cache))
            cache.write_text('{bad json')
            with self.assertRaises(ValueError):
                cached_selection(video, cache)
            update_selection_cache(video, path=cache)
            self.assertIsNone(cached_selection(video, cache))

    def test_valid_schedule_and_reject_duplicate_frames(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'selections.json'
            item = {'frame': 30, 'box': [10, 10, 50, 110]}
            path.write_text(json.dumps([item]))
            self.assertEqual(load_selections(path), {30: (10., 10., 50., 110.)})
            path.write_text(json.dumps([item, item]))
            with self.assertRaises(ValueError):
                load_selections(path)

    def test_invalid_schedule_schema(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'selections.json'
            for data in ({}, [{'frame': -1, 'box': [1, 1, 2, 2]}],
                         [{'frame': True, 'box': [1, 1, 2, 2]}],
                         [{'frame': 1}], [{'frame': 1, 'box': [0, 0, 0, 0]}]):
                path.write_text(json.dumps(data))
                with self.subTest(data=data), self.assertRaises(ValueError):
                    load_selections(path)


if __name__ == '__main__':
    unittest.main()

"""Conservative target policy, independent of the detector and video backend."""
from dataclasses import dataclass
from math import hypot, isfinite


def validate_box(box, width=None, height=None):
    if len(box) != 4 or not all(isfinite(v) for v in box):
        raise ValueError('Box must contain four finite coordinates')
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 and 0 <= y1 < y2):
        raise ValueError('Use 0 <= x1 < x2 and 0 <= y1 < y2')
    if width is not None and (x2 > width or y2 > height):
        raise ValueError('Box is outside the original frame')
    return tuple(float(v) for v in box)


def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def intersection(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))


def iou(a, b):
    overlap = intersection(a, b)
    return overlap / max(area(a) + area(b) - overlap, 1e-9)


@dataclass
class Person:
    track_id: int
    box: tuple
    confidence: float = 1.0
    white_fraction: float = 0.0
    blue_fraction: float = 0.0
    visual: tuple = ()
    lower_blue_fraction: float = 0.0


def crowded(person, detection_boxes):
    # Ignore the detection corresponding to this track, but inspect untracked people too.
    own = max(range(len(detection_boxes)),
              key=lambda j: iou(person.box, detection_boxes[j]), default=None)
    return any(j != own and intersection(person.box, box) / min(area(person.box), area(box)) > 0.4
               for j, box in enumerate(detection_boxes))


class Target:
    def __init__(self):
        self.target_id = None
        self.track_id = None
        self.box = None
        self.white_fraction = None
        self.blue_fraction = None
        self.visual = ()
        self.velocity = (0.0, 0.0)
        self.status = 'unselected'
        self.reason = 'Select the white athlete'
        self.clear_frames = 0
        self.appearance_grace = 0
        self.uncertain_frames = 0
        self.candidate_id = None
        self.candidate_frames = 0
        self.reidentify_allowed = False

    def select(self, box, people, detection_boxes):
        validate_box(box)
        matches = sorted(people, key=lambda p: iou(box, p.box), reverse=True)
        if not matches or iou(box, matches[0].box) < 0.5:
            raise ValueError('Selection must match a detected person (IoU >= 0.5)')
        if len(matches) > 1 and iou(box, matches[1].box) > 0.3:
            raise ValueError('Ambiguous selection; choose a clearer frame')
        person = matches[0]
        if person.confidence < 0.35 or crowded(person, detection_boxes):
            raise ValueError('Low confidence or overlapping people; choose a clearer frame')
        if self.target_id is None:
            self.target_id = 1
        self.track_id, self.box = person.track_id, person.box
        self.white_fraction = person.white_fraction
        self.blue_fraction = person.blue_fraction
        self.visual = person.visual
        self.velocity = (0.0, 0.0)
        self.status, self.reason = 'tracking', 'User confirmed target'
        self.clear_frames = 0
        self.appearance_grace = 0
        self.uncertain_frames = 0
        self.candidate_id = None
        self.candidate_frames = 0
        self.reidentify_allowed = False

    def _reidentify(self, people, detection_boxes):
        # The blue gi has a white back patch; require little blue on both torso and trousers.
        if not self.reidentify_allowed or not self.visual:
            return
        old = self.box
        diagonal = hypot(old[2] - old[0], old[3] - old[1])
        predicted = ((old[0] + old[2]) / 2 + self.velocity[0] * min(self.uncertain_frames, 8),
                     (old[1] + old[3]) / 2 + self.velocity[1] * min(self.uncertain_frames, 8))
        candidates = []
        for person in people:
            if person.track_id == self.track_id or person.confidence < 0.5 or crowded(person, detection_boxes):
                continue
            distance = hypot((old[0] + old[2] - person.box[0] - person.box[2]) / 2,
                             (old[1] + old[3] - person.box[1] - person.box[3]) / 2)
            motion_distance = hypot((person.box[0] + person.box[2]) / 2 - predicted[0],
                                    (person.box[1] + person.box[3]) / 2 - predicted[1])
            scale = area(person.box) / area(old)
            if (distance <= 0.75 * diagonal and 0.4 <= scale <= 2.5
                    and motion_distance <= 0.85 * diagonal
                    and len(person.visual) == len(self.visual)
                    and sum(a * b for a, b in zip(person.visual, self.visual)) >= 0.7
                    and person.white_fraction >= 0.2
                    and person.white_fraction > person.blue_fraction + 0.15
                    and abs(person.white_fraction - self.white_fraction) <= 0.45
                    and person.blue_fraction <= min(0.15, self.blue_fraction + 0.15)
                    and person.lower_blue_fraction <= 0.1):
                candidates.append(person)
        if len(candidates) != 1:
            self.candidate_id, self.candidate_frames = None, 0
            return
        candidate = candidates[0]
        if candidate.track_id == self.candidate_id:
            self.candidate_frames += 1
        else:
            self.candidate_id, self.candidate_frames = candidate.track_id, 1
        if self.candidate_frames >= 3:
            self.track_id, self.box = candidate.track_id, candidate.box
            self.visual = candidate.visual
            self.velocity = (0.0, 0.0)
            self.status, self.reason = 'tracking', 'New ID confirmed by motion and appearance'
            self.uncertain_frames = self.candidate_frames = self.clear_frames = 0
            self.candidate_id = None
            self.reidentify_allowed = False

    def update(self, people, detection_boxes, camera_cut=False):
        # shortcut: no learned ReID; only rejoin a nearby, visually consistent white gi after three clear frames.
        recovering = self.status == 'uncertain' and self.reason != 'Possible camera cut'
        if self.status != 'tracking' and not recovering:
            return None
        person = next((p for p in people if p.track_id == self.track_id), None)
        if camera_cut:
            self.status, self.reason = 'uncertain', 'Possible camera cut'
            self.reidentify_allowed = False
        elif person is None:
            self.status, self.reason = 'uncertain', 'Selected track is absent'
        elif person.confidence < 0.35:
            self.status, self.reason = 'uncertain', 'Weak person detection'
        else:
            old = self.box
            distance = hypot((old[0] + old[2] - person.box[0] - person.box[2]) / 2,
                             (old[1] + old[3] - person.box[1] - person.box[3]) / 2)
            scale = area(person.box) / area(old)
            visual_match = (bool(self.visual and person.visual)
                            and len(person.visual) == len(self.visual)
                            and sum(a * b for a, b in zip(person.visual, self.visual)) >= 0.85)
            overlap_continuity = (distance <= 0.5 * hypot(old[2] - old[0], old[3] - old[1])
                                  and 0.5 <= scale <= 2
                                  and person.confidence >= 0.5
                                  and person.white_fraction >= 0.25
                                  and person.white_fraction > person.blue_fraction + 0.15
                                  and person.blue_fraction <= 0.15
                                  and person.lower_blue_fraction <= 0.1
                                  and visual_match)
            if crowded(person, detection_boxes) and not overlap_continuity:
                # An opponent's sleeve can temporarily contaminate the jersey crop.
                if person.white_fraction >= 0.15 and person.white_fraction >= person.blue_fraction:
                    self.box = person.box
                self.status, self.reason = 'uncertain', 'Overlapping boxes; identity ambiguous'
                self.reidentify_allowed = True
                self.clear_frames = 0
                self.appearance_grace = 0
                self.candidate_id, self.candidate_frames = None, 0
                return None
            max_distance = (0.75 if recovering else 0.5) * hypot(old[2] - old[0], old[3] - old[1])
            min_scale, max_scale = (0.25, 4) if recovering else (0.5, 2)
            if person.blue_fraction >= 0.18 and person.blue_fraction >= person.white_fraction:
                self.status, self.reason = 'uncertain', 'Blue-dominant or merged person box'
            elif recovering and scale > 1.5 and person.white_fraction < 0.3:
                self.status, self.reason = 'uncertain', 'Possible merged person box'
            elif distance > max_distance or not min_scale <= scale <= max_scale:
                self.status, self.reason = 'uncertain', 'Abrupt position or size change'
            elif (abs(person.white_fraction - self.white_fraction) > 0.35
                  or abs(person.blue_fraction - self.blue_fraction) > 0.2):
                if recovering and self.appearance_grace < 5:
                    self.appearance_grace += 1
                    self.clear_frames = 0
                    if person.white_fraction >= 0.15 and person.white_fraction >= person.blue_fraction:
                        self.box = person.box
                    return None
                self.status, self.reason = 'uncertain', 'Jersey appearance changed'
            elif (recovering and self.visual and person.visual
                  and len(person.visual) == len(self.visual)
                  and sum(a * b for a, b in zip(person.visual, self.visual)) < 0.6):
                self.status, self.reason = 'uncertain', 'Visual appearance changed'
            else:
                self.appearance_grace = 0
                self.velocity = (0.6 * self.velocity[0] + 0.4 *
                                 ((person.box[0] + person.box[2] - old[0] - old[2]) / 2),
                                 0.6 * self.velocity[1] + 0.4 *
                                 ((person.box[1] + person.box[3] - old[1] - old[3]) / 2))
                self.box = person.box
                if recovering:
                    self.clear_frames += 1
                    if self.clear_frames < 3:
                        return None
                    self.status = 'tracking'
                self.reason = 'Same track with continuity checks'
                if person.visual:
                    self.visual = person.visual
                self.uncertain_frames = 0
                self.candidate_id, self.candidate_frames = None, 0
                self.reidentify_allowed = False
                return person.box
        if self.status == 'uncertain' and self.reason != 'Possible camera cut':
            self.uncertain_frames += 1
            self._reidentify(people, detection_boxes)
            if self.status == 'tracking':
                return self.box
            if self.status == 'uncertain' and self.uncertain_frames >= 45:
                self.status, self.reason = 'lost', 'Target not confirmed for 45 frames'
        return None


def csv_row(timestamp, target, box):
    return [f'{timestamp:.6f}', target.target_id if target.target_id is not None else '',
            *(box if box is not None else ['', '', '', '']), target.status,
            target.track_id if box is not None else '']

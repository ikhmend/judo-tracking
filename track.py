"""YOLOv8 person detection + ByteTrack, with explicit target confirmation."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

from target import Person, Target, area, crowded, csv_row, intersection, validate_box

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / '.target_selections.json'


def choose_device(requested):
    import torch
    available = torch.cuda.is_available()
    print(f'PyTorch={torch.__version__}; built CUDA={torch.version.cuda}; CUDA available={available}')
    if requested != 'cpu' and available:
        try:
            torch.ones(1, device='cuda').add_(1)
            torch.cuda.synchronize()
            print(f'GPU: {torch.cuda.get_device_name(0)}')
            return torch.device('cuda:0')
        except RuntimeError as exc:
            if requested == 'cuda':
                raise RuntimeError(f'CUDA probe failed: {exc}') from exc
            print(f'CUDA probe failed, using CPU: {exc}')
    elif requested == 'cuda':
        raise ValueError('CUDA is unavailable. Use --device cpu or install the CUDA PyTorch wheel.')
    print('Using CPU')
    return torch.device('cpu')


class Detector:
    def __init__(self, weights, device):
        if not weights.is_file():
            raise ValueError('YOLOv8 weights missing. Run setup.sh first.')
        from ultralytics import YOLO
        self.model = YOLO(str(weights.resolve()), task='detect')
        self.device = str(device)

    def detect(self, frame):
        import supervision as sv
        # Keep low-confidence people available to ByteTrack's association step.
        result = self.model.predict(frame, imgsz=640, conf=0.1, iou=0.45,
                                    classes=[0], device=self.device, verbose=False)[0]
        if not len(result.boxes):
            return sv.Detections.empty()
        boxes = result.boxes
        return sv.Detections(xyxy=boxes.xyxy.cpu().numpy(),
                             confidence=boxes.conf.cpu().numpy(),
                             class_id=boxes.cls.cpu().numpy().astype(int))


def jersey_fractions(frame, box):
    import cv2
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    torso = frame[int(y1 + height * 0.18):int(y1 + height * 0.7),
                  int(x1 + width * 0.1):int(x1 + width * 0.9)]
    if not torso.size:
        return 0.0, 0.0
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] < 90) & (hsv[:, :, 2] > 130)
    blue = ((hsv[:, :, 0] > 90) & (hsv[:, :, 0] < 135)
            & (hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 45))
    return float(white.mean()), float(blue.mean())


def lower_blue_fraction(frame, box):
    import cv2
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    lower = frame[int(y1 + height * 0.62):int(y1 + height * 0.97),
                  int(x1 + width * 0.18):int(x1 + width * 0.82)]
    if not lower.size:
        return 0.0
    hsv = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV)
    blue = ((hsv[:, :, 0] > 90) & (hsv[:, :, 0] < 135)
            & (hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 45))
    return float(blue.mean())


def visual_descriptor(frame, box):
    import cv2
    import numpy as np
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    torso = frame[int(y1 + h * .18):int(y1 + h * .7),
                  int(x1 + w * .1):int(x1 + w * .9)]
    if not torso.size:
        return ()
    gray = cv2.cvtColor(cv2.resize(torso, (32, 64)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    angle %= 180
    parts = []
    for ys in (slice(0, 32), slice(32, 64)):
        for xs in (slice(0, 16), slice(16, 32)):
            hist = np.histogram(angle[ys, xs], bins=9, range=(0, 180),
                                weights=magnitude[ys, xs])[0]
            parts.extend(hist / max(float(hist.sum()), 1e-9))
    feature = np.asarray(parts)
    norm = float(np.linalg.norm(feature))
    return tuple(feature / norm) if norm else ()


def athlete_boxes(frame, detections):
    # shortcut: dark uniforms without visible gi are ignored by the overlap guard; use a trained athlete classifier if this occurs often.
    return [tuple(float(v) for v in box)
            for box, confidence in zip(detections.xyxy, detections.confidence)
            if confidence >= 0.35 and max(jersey_fractions(frame, box)) >= 0.12]


def clear_white_candidate(people, boxes):
    return any(person.confidence >= 0.35 and person.white_fraction >= 0.18
               and person.white_fraction > person.blue_fraction + 0.1
               and not crowded(person, boxes) for person in people)


def load_selections(path):
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, list):
        raise ValueError('Selections JSON must be a list of {frame, box}')
    selections = {}
    for item in data:
        if not isinstance(item, dict) or set(item) != {'frame', 'box'}:
            raise ValueError('Each selection needs exactly frame and box')
        index = item['frame']
        if type(index) is not int or index < 0 or index in selections:
            raise ValueError('Selection frames must be unique nonnegative integers')
        selections[index] = validate_box(item['box'])
    return selections


def selection_cache(path=CACHE):
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise ValueError(f'Invalid selection cache {path}; use --reset-selection') from exc
    if not isinstance(data, dict):
        raise ValueError(f'Invalid selection cache {path}; use --reset-selection')
    return data


def cached_selection(video, path=CACHE):
    item = selection_cache(path).get(str(video.resolve()))
    if item is None:
        return None
    if not isinstance(item, dict):
        raise ValueError(f'Invalid selection cache {path}; use --reset-selection')
    stat = video.stat()
    if item.get('size') != stat.st_size or item.get('mtime_ns') != stat.st_mtime_ns:
        return None
    frame = item.get('frame')
    if type(frame) is not int or frame < 0:
        raise ValueError(f'Invalid selection cache {path}; use --reset-selection')
    return frame, validate_box(item.get('box', ()))


def update_selection_cache(video, frame=None, box=None, path=CACHE):
    try:
        data = selection_cache(path)
    except ValueError:
        if frame is not None:
            raise
        data = {}
    key = str(video.resolve())
    if frame is None:
        data.pop(key, None)
    else:
        stat = video.stat()
        data[key] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                     'frame': frame, 'box': list(validate_box(box))}
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temporary.replace(path)


def log_transition(writer, index, fps, target, before, selected=False):
    old_status, old_track, old_reason = before
    base = [index, f'{index / fps:.6f}', target.target_id, old_track if old_track is not None else '',
            target.track_id if target.track_id is not None else '']
    def write(event):
        writer.writerow([*base, event, target.status, target.reason])
    if old_track is not None and old_track != target.track_id:
        write('track_id_changed')
    if selected:
        write('selected')
    if target.status == 'lost' and old_status != 'lost':
        write('lost')
    elif target.status == 'tracking' and old_status in ('uncertain', 'lost'):
        write('recovered')
    elif target.status == 'uncertain' and (old_status != 'uncertain' or old_reason != target.reason):
        write('uncertain')


def snapshot(frame, people, output, index):
    import cv2
    path = output / f'frame_{index:08d}.jpg'
    preview = frame.copy()
    for person in people:
        x1, y1, x2, y2 = (int(v) for v in person.box)
        cv2.rectangle(preview, (x1, y1), (x2, y2), (160, 160, 160), 2)
        cv2.putText(preview, f'ID {person.track_id}', (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if path.exists():
        raise ValueError(f'Snapshot already exists: {path}; use a fresh output folder')
    if not cv2.imwrite(str(path), preview):
        raise RuntimeError(f'Cannot save {path}')
    print(f'Frame {index} ({frame.shape[1]}x{frame.shape[0]}): {path}', flush=True)
    return path


def prompt_selection(frame, people, boxes, output, index, target):
    snapshot(frame, people, output, index)
    if not sys.stdin.isatty():
        print('No terminal input: use --box or --selections on a new run.')
        return None
    while True:
        answer = input('White athlete: x1 y1 x2 y2 | id N | next N | continue: ').strip()
        if answer == 'continue':
            return None
        if answer.startswith('id ') and target.status != 'unselected':
            try:
                chosen_id = int(answer.split()[1])
                person = next(p for p in people if p.track_id == chosen_id)
                target.select(person.box, people, boxes)
                return None
            except (ValueError, IndexError, StopIteration) as exc:
                print(f'Choose a clear person ID shown on the saved frame: {exc}')
                continue
        if answer.startswith('next '):
            try:
                offset = int(answer.split()[1])
                if offset > 0:
                    return index + offset
            except (ValueError, IndexError):
                pass
            print('N must be a positive integer')
            continue
        try:
            box = validate_box(tuple(map(float, answer.replace(',', ' ').split())),
                               frame.shape[1], frame.shape[0])
            target.select(box, people, boxes)
            return None
        except ValueError as exc:
            print(exc)


def debug_observation(frame, detections, people, target):
    if target.box is None:
        return 'unselected'
    nearby = []
    mat_people = 0
    for box, confidence in zip(detections.xyxy, detections.confidence):
        if (confidence >= 0.35 and area(box) >= 0.035 * frame.shape[0] * frame.shape[1]
                and box[3] >= 0.75 * frame.shape[0]):
            mat_people += 1
        if (confidence >= 0.35 and area(box) >= 0.4 * area(target.box)
                and intersection(box, target.box) / min(area(box), area(target.box)) >= 0.2):
            nearby.append(box)
    if len(nearby) == 1 and mat_people == 1:
        white, blue = jersey_fractions(frame, nearby[0])
        if white >= 0.15 and blue >= 0.18:
            return 'merged_box_suspected'
    if target.status == 'tracking':
        return 'confirmed'
    if any(p.track_id == target.track_id for p in people):
        return 'track_present_identity_ambiguous'
    return 'nearby_detection_no_target_track' if nearby else 'no_detection_near_last_target'


def draw_debug(frame, detections, people, target, observation):
    import cv2
    preview = frame.copy()
    cv2.rectangle(preview, (0, 0), (min(preview.shape[1], 1000), 89), (0, 0, 0), -1)
    cv2.putText(preview, f'target {target.target_id} / track {target.track_id} / {target.status}',
                (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(preview, target.reason, (8, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(preview, observation, (8, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    for box, confidence in zip(detections.xyxy, detections.confidence):
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 165, 255), 1)
        cv2.putText(preview, f'det {confidence:.2f}', (x1, max(y1 - 20, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 165, 255), 1)
    for person in people:
        x1, y1, x2, y2 = (int(v) for v in person.box)
        cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 220, 0), 1)
        cv2.putText(preview, f'track {person.track_id}', (x1, max(y1 - 5, 28)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 220, 0), 2)
    if target.status == 'tracking' and target.box is not None:
        x1, y1, x2, y2 = (int(v) for v in target.box)
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 220, 0), 3)
    return preview


def run(args):
    import cv2
    if args.video is None or args.output is None:
        raise ValueError('--video and --output are required')
    if not args.video.is_file() or args.video.suffix.lower() != '.mp4':
        raise ValueError('--video must be an existing local MP4')
    if args.reset_selection:
        update_selection_cache(args.video)
    selections = load_selections(args.selections)
    if args.box is not None:
        if args.start_frame in selections:
            raise ValueError('--box conflicts with a selection at --start-frame')
        selections[args.start_frame] = validate_box(args.box)
    cached = None if selections or args.reset_selection else cached_selection(args.video)
    if cached is not None:
        selections[cached[0]] = cached[1]
        print(f'Reusing saved target selection at frame {cached[0]}')
    args.output.mkdir(parents=True, exist_ok=True)
    paths = [args.output / name for name in ('tracked.mp4', 'tracking.csv', 'events.csv')]
    selection_output = args.output / 'confirmed_selections.json'
    debug_paths = [(args.output / f'debug_{start:g}_{end:g}.mp4', start, end)
                   for start, end in args.debug_range]
    debug_csv = args.output / 'debug_frames.csv'
    if args.extract_frame is None and any(path.exists() for path in
                                          [*paths, selection_output, *[p for p, _, _ in debug_paths],
                                           *([debug_csv] if debug_paths else [])]):
        raise ValueError('Output files already exist; use a fresh --output folder')
    capture = cv2.VideoCapture(str(args.video))
    writer = None
    debug_writers = {}
    debug_file = None
    try:
        if not capture.isOpened():
            raise ValueError('OpenCV could not open the MP4')
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('Video FPS is invalid')
        if args.extract_frame is not None:
            capture.set(cv2.CAP_PROP_POS_FRAMES, args.extract_frame)
            ok, frame = capture.read()
            if not ok:
                raise ValueError('Requested frame could not be decoded')
            snapshot(frame, [], args.output, args.extract_frame)
            return
        if frame_count > 0 and any(index >= frame_count for index in selections):
            raise ValueError('Selection frame is beyond the end of the video')
        if frame_count > 0 and args.start_frame >= frame_count:
            raise ValueError('--start-frame is beyond the end of the video')
        import numpy as np
        import supervision as sv
        device = choose_device(args.device)
        detector = Detector(args.weights, device)
        tracker = sv.ByteTrack(track_activation_threshold=0.25, frame_rate=max(1, round(fps)))
        target = Target()
        next_prompt = args.start_frame if not selections else None
        interactive = args.interactive or not selections
        previous_thumbnail = None
        index = 0
        used_selections = []
        first_selection_saved = cached is not None
        if debug_paths:
            debug_file = debug_csv.open('x', newline='')
            debug_rows = csv.writer(debug_file)
            debug_rows.writerow(['frame', 'timestamp', 'target_id', 'target_track_id',
                                 'tracking_status', 'association_reason', 'observation',
                                 'target_track_present', 'detections', 'tracks'])
        with paths[1].open('x', newline='') as csv_file, paths[2].open('x', newline='') as event_file:
            rows, events = csv.writer(csv_file), csv.writer(event_file)
            rows.writerow(['timestamp', 'target_id', 'x1', 'y1', 'x2', 'y2', 'tracking_status', 'track_id'])
            events.writerow(['frame', 'timestamp', 'target_id', 'previous_track_id', 'track_id',
                             'event', 'tracking_status', 'reason'])
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if writer is None:
                    h, w = frame.shape[:2]
                    if w % 2 or h % 2:
                        raise ValueError('MP4 encoder needs even frame dimensions')
                    writer = cv2.VideoWriter(str(paths[0]), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    if not writer.isOpened():
                        raise RuntimeError('MP4 writer failed to open')
                elif frame.shape[:2] != (h, w):
                    raise ValueError('Video frame dimensions changed')
                thumbnail = cv2.resize(frame, (64, 36)).astype(np.float32) / 255
                # shortcut: frame difference misses subtle cuts, add shot detection after real-video calibration.
                cut = previous_thumbnail is not None and float(np.abs(thumbnail - previous_thumbnail).mean()) > 0.22
                previous_thumbnail = thumbnail
                detections = detector.detect(frame)
                # Weak detections and people without a visible gi do not veto an athlete selection.
                boxes = athlete_boxes(frame, detections)
                tracks = tracker.update_with_detections(detections)
                people = [Person(int(track_id), tuple(float(v) for v in box), float(confidence),
                                 *jersey_fractions(frame, box), visual_descriptor(frame, box),
                                 lower_blue_fraction(frame, box))
                          for box, confidence, track_id in zip(tracks.xyxy, tracks.confidence, tracks.tracker_id)]
                before = (target.status, target.track_id, target.reason)
                box = target.update(people, boxes, cut)
                log_transition(events, index, fps, target, before)
                changed = before != (target.status, target.track_id, target.reason)
                if changed:
                    print(f'{index / fps:.3f}s: {target.status}: {target.reason}', flush=True)
                    if target.status == 'tracking':
                        next_prompt = None
                    elif interactive and (target.status == 'lost' or target.reason == 'Possible camera cut'):
                        upcoming = min((frame for frame in selections if frame > index), default=None)
                        next_prompt = (upcoming + 1 if upcoming is not None and upcoming - index <= 30
                                       else index + 10)  # Let a lost target or camera transition settle.
                if index in selections:
                    validate_box(selections[index], w, h)
                    before = (target.status, target.track_id, target.reason)
                    target.select(selections[index], people, boxes)
                    next_prompt = None
                    used_selections.append({'frame': index, 'box': list(selections[index])})
                    log_transition(events, index, fps, target, before, selected=True)
                    if not first_selection_saved:
                        update_selection_cache(args.video, index, selections[index])
                        first_selection_saved = True
                    box = target.box
                elif (interactive and next_prompt is not None and index >= next_prompt
                      and clear_white_candidate(people, boxes) and target.status != 'tracking'):
                    csv_file.flush()
                    event_file.flush()
                    before = (target.status, target.track_id, target.reason)
                    next_prompt = prompt_selection(frame, people, boxes, args.output, index, target)
                    if target.status == 'tracking':
                        box = target.box
                        used_selections.append({'frame': index, 'box': list(box)})
                        log_transition(events, index, fps, target, before, selected=True)
                        if not first_selection_saved:
                            update_selection_cache(args.video, index, box)
                            first_selection_saved = True
                rows.writerow(csv_row(index / fps, target, box))
                for path, start, end in debug_paths:
                    if start <= index / fps < end:
                        observation = debug_observation(frame, detections, people, target)
                        target_present = any(p.track_id == target.track_id for p in people)
                        if path not in debug_writers:
                            debug_writers[path] = cv2.VideoWriter(
                                str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                            if not debug_writers[path].isOpened():
                                raise RuntimeError(f'Debug MP4 writer failed: {path}')
                        debug_writers[path].write(draw_debug(frame, detections, people, target,
                                                             observation))
                        raw_debug = []
                        for detected_box, confidence in zip(detections.xyxy, detections.confidence):
                            white, blue = jersey_fractions(frame, detected_box)
                            raw_debug.append({'box': [round(float(v), 2) for v in detected_box],
                                              'confidence': round(float(confidence), 4),
                                              'upper_white': round(white, 4),
                                              'upper_blue': round(blue, 4),
                                              'lower_blue': round(lower_blue_fraction(frame, detected_box), 4)})
                        debug_rows.writerow([
                            index, f'{index / fps:.6f}', target.target_id, target.track_id,
                            target.status, target.reason, observation, target_present,
                            json.dumps(raw_debug),
                            json.dumps([{'id': p.track_id, 'box': [round(v, 2) for v in p.box],
                                         'confidence': round(p.confidence, 4),
                                         'upper_white': round(p.white_fraction, 4),
                                         'upper_blue': round(p.blue_fraction, 4),
                                         'lower_blue': round(p.lower_blue_fraction, 4)}
                                        for p in people])])
                if box is not None:
                    x1, y1, x2, y2 = (int(v) for v in box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
                    cv2.putText(frame, f'Target {target.target_id} / track {target.track_id}',
                                (x1, max(18, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2)
                label = (f'{index / fps:.2f}s | target {target.target_id} | '
                         f'track {target.track_id if box is not None else "?"} | {target.status}')
                cv2.rectangle(frame, (0, 0), (min(w, 960), 66), (0, 0, 0), -1)
                cv2.putText(frame, label, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(frame, target.reason, (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                writer.write(frame)
                index += 1
                if index % 100 == 0:
                    csv_file.flush()
                    event_file.flush()
                    print(f'Processed {index}/{frame_count or "?"} frames', flush=True)
            if not index:
                raise ValueError('No video frames decoded')
            if frame_count > 0 and index < frame_count:
                raise RuntimeError(f'Decode ended early at {index}/{frame_count}; outputs are partial')
            missed = set(selections) - {item['frame'] for item in used_selections}
            if missed:
                raise ValueError(f'Selections were not reached: {sorted(missed)}')
        if debug_file is not None:
            debug_file.flush()
        with selection_output.open('x', encoding='utf-8') as handle:
            json.dump(used_selections, handle, indent=2)
        print(f'Done: {index} frames -> {args.output}')
        if not used_selections:
            print('No target was confirmed: all tracking rows are unselected.')
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        for debug_writer in debug_writers.values():
            debug_writer.release()
        if debug_file is not None:
            debug_file.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--weights', type=Path, default=ROOT / 'weights/yolov8n.pt')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--check-device', action='store_true')
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--box', nargs=4, type=float, metavar=('X1', 'Y1', 'X2', 'Y2'))
    parser.add_argument('--selections', type=Path, help='JSON list of explicit frame/box selections')
    parser.add_argument('--reset-selection', action='store_true', help='Forget saved initial target for this video')
    parser.add_argument('--interactive', action='store_true', help='Prompt for reselection after uncertainty')
    parser.add_argument('--extract-frame', type=int, help='Save a frame JPEG and exit; no model needed')
    parser.add_argument('--debug-range', nargs=2, type=float, action='append', default=[],
                        metavar=('START_SECONDS', 'END_SECONDS'),
                        help='Write raw detection/tracker overlays and per-frame debug CSV for this interval')
    args = parser.parse_args()
    if args.start_frame < 0 or (args.extract_frame is not None and args.extract_frame < 0):
        parser.error('Frame indices must be nonnegative')
    if any(not (math.isfinite(a) and math.isfinite(b) and 0 <= a < b)
           for a, b in args.debug_range):
        parser.error('Each --debug-range needs finite 0 <= start < end seconds')
    if len({f'{a:g}_{b:g}' for a, b in args.debug_range}) != len(args.debug_range):
        parser.error('Duplicate --debug-range intervals')
    try:
        if args.check_device:
            choose_device(args.device)
        else:
            run(args)
    except (ValueError, RuntimeError, OSError, ImportError, EOFError, TypeError) as exc:
        parser.exit(1, f'Error: {exc}\nInstall with setup.sh; use a fresh output folder after a failed run.\n')


if __name__ == '__main__':
    main()

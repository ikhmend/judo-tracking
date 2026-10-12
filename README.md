# Жудогийн target tracking — эхний хувилбар

Локал MP4 дээр хэрэглэгчийн сонгосон **цагаан өмсгөлтэй тамирчныг** YOLOv8 person detection + ByteTrack ашиглан дагана. Үндсэн боловсруулалт GUI, DISPLAY, WSLg шаардахгүй. Мэх, оноо, давуу/сул тал, сарын ахиц дүгнэхгүй.

## Загвар, tracker-ийн сонголт

- **YOLOv8n, COCO pretrained `yolov8n.pt`**: жижиг загварыг 640 оролтоор ажиллуулж, COCO-ийн `person` (class 0)-ийг л авна. Box, confidence болон class-ийг Ultralytics-ийн API-аас авч өмнөх tracker-т ижил `supervision.Detections` хэлбэрээр өгнө. [YOLOv8 албан ёсны documentation](https://docs.ultralytics.com/models/yolov8/), [predict API](https://docs.ultralytics.com/modes/predict/).
- **ByteTrack**: өндөр болон бага confidence-тэй detection-уудыг хөдөлгөөн, box association-аар холбоно. `supervision==0.25.0`-ийн tracker болон target identity дүрэм хэвээр; Ultralytics-ийн өөрийн tracker ашиглахгүй. [ByteTrack албан ёсны repository](https://github.com/FoundationVision/ByteTrack), [Supervision 0.25.0 tracker documentation](https://supervision.roboflow.com/0.25.0/trackers/).
- **PyTorch 2.5.1 + torchvision 0.20.1, Python 3.10–3.12**: төслийн virtual environment-ийн тогтоосон хувилбар. Python 3.14-ийг энэ dependency багцад ашиглахгүй. CPU болон CUDA wheel-ийн командыг [PyTorch албан ёсны суулгах заавраас](https://pytorch.org/get-started/previous-versions/#v251) авсан. YOLOv8-ийн CPU inference-ийг бодит видео дээр хараахан шалгаагүй.

Person detector нь шүүгч, үзэгч, тамирчны үүргийг ялгахгүй. Хэрэглэгч зөв тамирчны box-ийг сонгож target тогтооно. Бусад хүн target болж автоматаар сонгогдохгүй. Цагаан/хөх өмсгөлийн эзлэх хувийг appearance болон давхцлын туслах дохио болгоно; өнгөөр identity сонгохгүй. Өмсгөлийн өнгө бараг харагдахгүй detection-ийг давхцлын шалгалтаас хасна, ингэснээр ард зогссон шүүгч тамирчныг сонгоход саад болохгүй.

## WSL дээр суулгах

Ubuntu WSL terminal-д:

```bash
cd /home/lenovo/projects/judo-tracking
sudo apt update
sudo apt install -y python3-venv git curl libgl1 libglib2.0-0
python3 --version
```

Python 3.10–3.12 байгаа бол:

```bash
PYTHON=python3 bash setup.sh
source .venv/bin/activate
python track.py --check-device
```

Python 3.14 зэрэг өөр хувилбартай бол Python 3.11-ийг тусад нь суулгана. Жишээ нь [uv-ийн албан ёсны суулгалт](https://docs.astral.sh/uv/getting-started/installation/) болон [Python удирдах заавар](https://docs.astral.sh/uv/guides/install-python/)-ыг ашиглаж болно:

```bash
curl -LsSf https://astral.sh/uv/install.sh -o /tmp/judo-uv-install.sh
sh /tmp/judo-uv-install.sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.11
PYTHON="$(uv python find 3.11)" bash setup.sh
source .venv/bin/activate
```

`setup.sh` эхлээд NVIDIA GPU харагдах эсэхийг шалгана. Дараа нь `.venv`, dependencies, Ultralytics-ийн албан ёсны `yolov8n.pt` weight татаж, device probe болон тест ажиллуулна. Анхдагч суулгалт CPU wheel ашиглана. Өмнөх YOLOv6 `vendor/` болон `weights/yolov6n.pt` байсан бол устгах шаардлагагүй; шинэ код тэдгээрийг ашиглахгүй. `opencv-python` нь Supervision-ийн dependency учраас ашигласан; код window нээхгүй.

GPU ашиглах бол Windows NVIDIA driver, WSL-ийн GPU хандалт болон CUDA 12.4 wheel-д нийцэх driver хэрэгтэй. GPU харагдаж байгаа орчинд:

```bash
PYTHON=python3 TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 bash setup.sh
source .venv/bin/activate
python track.py --check-device --device cuda
```

Өмнө CPU wheel суулгасан бол үүнийг `.venv` дотор солино:

```bash
python -m pip install --force-reinstall torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124
```

`--device auto` нь CUDA байгаа эсэх болон жижиг tensor operation ажиллах эсэхийг шалгаж CPU руу fallback хийнэ. `--device cuda` GPU ажиллахгүй бол ил тод алдаа өгнө. `--device cpu` CPU-г албадана. CUDA Toolkit тусад нь суулгах нь CPU хувилбарт шаардлагагүй. Weight нь Python checkpoint тул зөвхөн итгэмжлэгдсэн албан ёсны `.pt` ашиглана.

## Target сонгож ажиллуулах

Windows-ийн `C:\Videos\match.mp4` нь WSL дээр `/mnt/c/Videos/match.mp4` болно. Output болгонд шинэ хавтас хэрэглэнэ; өмнөх үр дүнг дарж бичихгүй.

**Terminal-аар сонгох:**

```bash
python track.py --video /mnt/c/Videos/match.mp4 --output outputs/run01
```

Эхний person track-тай кадрыг `frame_00000000.jpg` зэрэг нэрээр output хавтаст хадгалж, зам болон зургийн хэмжээг хэвлэнэ. Хэрэв хэрэгтэй кадр хожуу байвал `--start-frame 120` ашиглана. Windows зураг үзэгчээр файлыг нээж, цагаан тамирчны бүтэн биед ойролцоо box сонгоод terminal-д **`x1 y1 x2 y2`** оруулна. Координат нь original frame-ийн пиксел; зүүн дээд булан `(0, 0)`, x баруун, y доош өснө. Зураг resize хийж харвал координатыг original хэмжээ рүү шилжүүлнэ.

```text
120 80 350 600
```

Сонголт detection-той IoU >= 0.5 таарч, confidence >= 0.35, хүн давхцаагүй байх ёстой. Тохиромжгүй бол `next 30` гэж дараагийн 30 кадрын дараа зураг гаргуулна. `continue` гэвэл сонголтгүй/алдагдсан төлөвөөр боловсруулалтыг үргэлжлүүлнэ. Түр `uncertain` үед дахин сонголт асуухгүй; 45 кадр батлагдахгүй өнгөрвөл `lost` болно. Камер солигдвол дахин сонголт шаардлагатай. Prompt-ыг зөвхөн цагаан өмсгөлтэй тод, давхцаагүй candidate гарсан үед нэг удаа харуулна; `next N` гэвэл N кадрын дараа дахин шалгана. Шууд terminal байхгүй бол `--box` эсвэл `--selections` ашиглана; сонголтгүй боловсруулбал бүх мөр `unselected` болно.

Эхний баталсан сонголт `.target_selections.json` файлд **сонгомогц** хадгалагдана. Ижил видео зам, хэмжээ, өөрчлөгдсөн хугацаатай дараагийн run-д `--box`/`--selections` өгөөгүй бол автоматаар ашиглана. Шинээр сонгох бол `--reset-selection` нэмнэ. Энэ нь зөвхөн тухайн видеоны хадгалсан сонголтыг арилгана; `--start-frame 300`-аар шинэ сонголтын кадрыг зааж болно.

**GUI, terminal input шаардлагагүй координатын хувилбар:**

```bash
# Эхлээд кадр гаргана; YOLOv8 model/PyTorch хэрэггүй, OpenCV хэрэгтэй.
python track.py --video /mnt/c/Videos/match.mp4 \
  --output outputs/preview30 --extract-frame 30

# Зургийг Windows талаас нээх хавтас:
explorer.exe "$(wslpath -w "$PWD/outputs/preview30")"

# Жишээ координатыг өөрийн видеонд тааруулж солино.
python track.py --video /mnt/c/Videos/match.mp4 \
  --output outputs/run02 --start-frame 30 --box 120 80 350 600 --device cpu
```

Видео эхнээсээ бүтнээрээ боловсруулагдана. Сонгосон кадраас өмнө төлөв `unselected`; timestamp нь эх видеоноос тооцогдоно. Frame index нь **0-ээс** эхэлнэ. `--box` ашиглах үед тасралтгүй автомат боловсруулна; lost үед terminal-аар асуулгахыг хүсвэл `--interactive` нэмнэ.

## Target дахин сонгох

Target алдагдвал хамгийн ойрын хүн эсвэл ижил өнгийн өмсгөлөөр дангаар нь өөр ID авахгүй. Давхцал ажиглагдсаны дараа хэд хэдэн бие даасан шинжээр шинэ ID батлагдвал шилжилтийг `events.csv`-д ил тэмдэглэнэ. Батлагдахгүй, эсвэл камер солигдвол хэрэглэгч цагаан тамирчныг дахин сонгоно.

Terminal горимд `next N`-ээр илүү тод кадр хүртэл алгасаж, тэр кадрын box-ийг оруулж болно. Дахин сонгох үед зурган дээрх ID-г `id 34` хэлбэрээр оруулж мөн болно; ID-г хэрэглэгч өөрөө батална. `continue` сонгосон бол дахин асуухгүй; дараах JSON аргаараа шинэ run ажиллуулна.

Terminal-гүй үед `events.csv`-ийн frame-ийг үзэж, тэр болон дараах тод кадрыг `--extract-frame`-аар гаргаад `selections.json` бэлдэнэ:

```json
[
  {"frame": 30, "box": [120, 80, 350, 600]},
  {"frame": 900, "box": [500, 100, 740, 620]}
]
```

```bash
cp selections.example.json selections.json
# selections.json дахь frame, box-ийг өөрийн видеонд тааруулж засна.
python track.py --video /mnt/c/Videos/match.mp4 \
  --output outputs/run03 --selections selections.json --device auto
```

JSON-ийн сонголт бүр хүний баталсан сонголт байх ёстой. Буруу/давхардсан frame, image-ээс гадуурх box, таарах detection байхгүй бол алдаа өгнө. `--selections` эсвэл `--box` нь хадгалсан эхний сонголтыг энэ run-д орлоно. Бүх баталсан сонголт амжилттай дууссан run-ийн `confirmed_selections.json`-д үлдэнэ.

## Гаралт ба төлөв

| Файл | Агуулга |
|---|---|
| `tracked.mp4` | Тогтвортой target ID, тухайн үеийн ByteTrack ID, box, timestamp, төлөв |
| `tracking.csv` | Кадр бүрд нэг мөр: `timestamp,target_id,x1,y1,x2,y2,tracking_status,track_id` |
| `events.csv` | `track_id_changed`, `lost`, `recovered`, `uncertain`, `selected` үйл явдал ба timestamp |
| `confirmed_selections.json` | Хэрэглэгчийн баталсан сонголтууд |
| `.target_selections.json` | Видеоны эхний сонголтын run хоорондын cache (төслийн үндсэн хавтас) |
| `debug_<start>_<end>.mp4`, `debug_frames.csv` | `--debug-range` өгсөн үед raw person detection, confidence, бүх tracker ID, target-ийн шийдвэр |

`tracking`, `lost`, `uncertain`, `unselected` төлөвтэй. `target_id=1` нь энэ видеонд хэрэглэгчийн сонгосон нэг тамирчны дугаар; ByteTrack-ийн түр `track_id` солигдсон ч хэвээр. `lost/uncertain` үед **box зурахгүй, CSV-ийн дөрвөн координат ба одоогийн track_id хоосон** байна. ID өөрчлөгдвөл хуучин/шинэ track ID-г `events.csv`-д тэмдэглэнэ.

```csv
timestamp,target_id,x1,y1,x2,y2,tracking_status,track_id
1.000000,1,120.0,80.0,350.0,600.0,tracking,7
1.033333,1,,,,,uncertain,
```

Timestamp нь `frame_index / FPS`, нэгж нь секунд. Энэ хувилбар constant frame rate MP4 гэж үзнэ; variable frame rate видео бол эхлээд constant FPS болгон хөрвүүлнэ. Output MP4 нь original хэмжээ/FPS-тэй, **дуу агуулахгүй**. Алдаа эсвэл Ctrl+C үед writer/capture хаагдана, дутуу файлууд үлдэж болно; амжилтын `Done` гараагүй үр дүнг бүрэн гэж үзэхгүй.

Барьц/мэхийн үеийг шалгахдаа секундын хоёр интервал өгнө:

```bash
python track.py --video /mnt/c/Users/Lenovo/Videos/match.mp4 \
  --output outputs/debug_run --selections selections.match.reviewed.json \
  --device cpu --debug-range 29 50 --debug-range 105 115
```

`debug_29_50.mp4`, `debug_105_115.mp4`-д улбар шар нимгэн box нь шүүлтээс өмнөх **бүх person detection** (`det` confidence), цэнхэр box нь tracker-ийн бүх ID, ногоон box нь зөвхөн батлагдсан target. `debug_frames.csv`-ийн мөр бүр raw detection/track box-уудыг JSON-аар, `association_reason`, `target_track_present`, `observation`-ийг хадгална. `merged_box_suspected` нь нэг том box-д цагаан/хөх шинж зэрэг илэрсэн **сэжиг**; хүний дүрсний шалгалтаар батална. Шинэ output хавтас хэрэглэнэ.

## Identity хамгаалах логик ба хязгаар

Сонгосон track ID кадрт байхгүй бол эхлээд `uncertain`; 45 кадр батлагдахгүй өнгөрвөл `lost`. Box-ууд давхцсан ч ижил track ID, тогтвортой байрлал/хэмжээ, дүрсний шинж, цагаан өмсгөл болон хөх бус доод өмсгөлийн шинж **хамтдаа** нийцвэл tracking үргэлжилнэ. Эдгээр нотолгоо дутвал `Overlapping boxes; identity ambiguous`; хөх давамгайлсан эсвэл давхцлын дараа томорсон нэг box-ийг батлахгүй. Confidence сулрах, огцом байрлал/хэмжээ өөрчлөгдөх, өмсгөлийн шинж эсвэл grayscale дүрсний descriptor өөрчлөгдөх, camera cut илрэхэд мөн `uncertain`. Confidence 0.1–0.35 detection-ууд ByteTrack-д хэвээр очно, гэхдээ сул duplicate/body-part detection-ийг өөр хүн гэж үзэхгүй. Давхцлын дараах шинэ ID-г зөвхөн өмнөх байрлал, хөдөлгөөний таамаг, хэмжээ, дүрсний descriptor, дээд/доод өмсгөлийн шинж **хамтдаа** болон гурван дараалсан тод кадр таарсан үед батална; бусад үед хэрэглэгч дахин сонгоно.

Эдгээр босго нь эхний хамгаалалт; жудогийн тэврэлт/барьц үед олон удаа uncertain болж болно. Camera cut нь жижигрүүлсэн хоёр кадрын дундаж ялгаагаар хэмжигдэх heuristic. Зөөлөн cut, detector хоёр хүнийг нэг box болгох, ижил өнгийн хүн нэг байрлалд орж ирэх зэрэг тохиолдолд tracker-ийн ID switch-ийг бүрэн илрүүлэх баталгаа байхгүй. Шинэ ID сэргээх нь дүрэмд суурилсан heuristic; сурсан ReID эсвэл pose identity баталгаажуулалт энэ шатанд байхгүй. Тайлангийн хөдөлгөөний шинжилгээнд ашиглахын өмнө бодит видео дээр хүний review шаардлагатай.

## Шалгалт

```bash
python -m unittest discover -s tests -v
python -m py_compile track.py target.py tests/test_target.py
bash -n setup.sh
```

32 unit test: YOLOv8 detection-ийг tracker-ийн оролт руу хөрвүүлэх, тогтвортой target ID, түр track ID, хөдөлгөөний таамаг/дүрсний шинжийн gate, сонголтын хадгалалт/reset, event log, давхцал, prompt-ийн тод кадр, camera cut, координат/JSON validation, CPU fallback. Detector/Device тестүүд mock ашиглана; бодит YOLOv8 inference/GPU probe биш.

**YOLOv8 руу шилжилтийн төлөв:** Зөвхөн detector adapter, default weight болон суулгах dependency өөрчлөгдсөн. ByteTrack ба target identity логик хэвээр. Энэ орчинд PyPI DNS ажиллахгүй байсан тул `ultralytics==8.4.142` болон `yolov8n.pt`-ийг татаж чадаагүй; доорх бодит видеоны тоонууд нь **хуучин YOLOv6 baseline** бөгөөд YOLOv8-ийн үр дүн биш. Сүлжээтэй WSL орчинд `bash setup.sh` ажиллуулсны дараа шинэ output хавтаст ижил MP4-ээ дахин боловсруулж, `tracking.csv`, `events.csv`, MP4-ийн хайрцгуудыг baseline-тай харьцуулна.

**Анхны орчинд шалгасан зүйл:** WSL2, Python 3.14.4; `nvidia-smi`, `/dev/dxg`, WSL NVIDIA binary олдоогүй. Python virtual environment `.venv-check` үүсгэж, 14 тест, Python syntax, CLI help, shell syntax шалгасан. Тэр үед package татах DNS ажиллаагүй, Python 3.11 байхгүй тул dependencies/model weights суулгах, inference ажиллуулах боломжгүй байсан. `.venv-check` нь зөвхөн стандарт сангийн тестэд зориулсан.

**Dependencies суулгасны дараах бодит видеоны шалгалт:** Хэрэглэгч Python 3.11 `.venv`, YOLOv6 weight болон dependencies суулгаж, `/mnt/c/Users/Lenovo/Videos/match.mp4` видеог өгсний дараа CPU inference ажиллуулсан. Видео 1250×644, 30 FPS, 3954 кадртай. 301-р кадрт цагаан тамирчны доод биеийг confidence 0.101-тэй давхар detection гэж гаргаж, хуурамч overlap үүсгэснийг зассан. Бага confidence detection ByteTrack-д хэвээр очих боловч overlap guard confidence >= 0.35 detection-уудыг ашиглана.

300–360-р кадрын 61 кадрыг тусдаа MP4 болгон гаргаж, CLI-аар detector→tracker→MP4/CSV боловсруулалт шалгасан: original 10.000–10.467 секундэд target нэг ID-тай tracking; 10.500–12.000 секундэд хоёр тамирчны box давхцах дохиогоор uncertain түгжигдсэн. MP4 61 кадртай, CSV 61 мөртэй, uncertain үед координат хоосон, ID өөрчлөгдөөгүйг assert-аар баталсан. 301 болон 315-р кадрыг зургаар нягталсан. Энэ нь дахин encode хийсэн богино хэсгийн шалгалт; **бүтэн видеоны ID switch/lost интервалуудыг хараахан хянаагүй**. Android emulator-ийн MP4-үүдийг жудогийн validation гэж тооцоогүй.

**Сайжруулалтын дараах шалгалт:** `run02`-ийн бүтэн CSV-д 3954 кадрын 16 нь tracking, 3638 нь uncertain (`run02` нь хуучин кодоор гарсан). Видеоны 300–600-р кадрыг дахин encode хийсэн 301 кадрын хэсэгт сонголтгүй бол камер солигдсоноос хойш шинэ ID-г авахгүй. 300, 400, 510-р кадарт цагаан тамирчныг зургаар нягталж хүн баталсан `selections.match.reviewed.json`-ийг ашигласан шалгалтад **172 tracking, 75 uncertain, 54 lost** кадр гарсан. Үзэх жишээ: `outputs/validation_10_20/tracked.mp4`, түүнд харгалзах CSV/event мөн тэр хавтаст байна. Жишээний CSV хугацаа 0-ээс эхэлдэг; эх бичлэгийн хугацааг авахын тулд 10 секунд нэмнэ. Энэ нь зөвхөн 10 секундийн хэсэг; бүтэн видео болон бүх ID солигдолтыг баталгаажуулсан гэсэн үг биш.

1086-р кадрт цагаан тамирчныг ард байгаа шүүгчтэй давхцсан гэж буруу татгалзаж байсныг өмсгөлийн өнгийг зөвхөн давхцлын шүүлтэд ашиглаж зассан. Эх MP4-ийн тэр кадрт YOLOv6-ийн detection-уудыг дахин шалгахад цагаан/хөх тамирчны box хэвээр, шүүгчийн box шүүлтээс хасагдаж, цагаан тамирчны сонголт `tracking` болсон. Энэ засвар аль хэдийн ажиллаж буй `run03` process-д үйлчлэхгүй; шинэ output хавтастай дахин ажиллуулах хэрэгтэй.

**ID тасралтын дараах шалгалт:** Эх MP4-ийн 300–650-р кадрыг YOLOv6 + ByteTrack-тай дахин боловсруулж, 300, 400, 510-р кадрын баталсан box-уудыг ашиглахад 351 кадраас 219 `tracking`, 132 `uncertain`, 0 `lost` гарсан. Давхцлын дараа 571, 615-р кадрт шинэ ID-г баталсан; хоёр кадрт box цагаан тамирчин дээр байгааг зургаар нягталсан. Мөн энэ хэсгийг дахин encode хийсэн MP4-ээр CLI-аар ажиллуулахад 351 кадраас 215 `tracking`, 136 `uncertain`, 0 `lost`; 610-р эх кадрын шинэ ID цагаан тамирчинд байгааг MP4-ээр шалгасан. Үр дүн: `outputs/validation_reid_10_21/tracked.mp4`, `tracking.csv`, `events.csv`. Жишээний CSV хугацаанд эх видеоны 10 секундийг нэмнэ. Эдгээр нь бүтэн бичлэгийн баталгаа биш.

**Тогтвортой target ID-тай шинэ хувилбар:** Дээрх 351 кадрын клипийг CLI-аар дахин ажиллуулахад 215 `tracking`, 136 `uncertain`, 0 `lost`. `target_id=1` бүх 351 мөрөнд хэвээр; `track_id` нь 2 → 7 → 34 → 59 болсон. Эх бичлэгийн 400, 510-р кадрт хүний баталсан сонголт, 610-р кадрт байрлал/хөдөлгөөн/дүрс/өмсгөлийн шинжээр баталсан автомат сэргэлт `events.csv`-д тус тус тэмдэглэгдсэн. `outputs/validation_stable_10_21/tracked.mp4`-ийн эх бичлэгийн 325, 400, 450, 510, 550, 610, 640-р кадрын орчмын box-уудыг зургаар шалгахад цагаан тамирчин дээр байсан. MP4 351 кадртай; CSV-д tracking үеийн box ба track ID бүрэн, uncertain үеийнх хоосон байгааг шалгасан. Эх видеоны 300-р кадрт хэрэглэгчийн баталсан анхны box-ийг `.target_selections.json`-д хадгалсан. Энэ нь бүтэн 3954 кадрыг шалгасан гэсэн үг биш.

Шүүгч ойр, тамирчид удаан барьцалдсан эх бичлэгийн 1080–1350-р кадрын CLI шалгалтад 6 `unselected`, 23 `tracking`, 132 `uncertain`, 110 `lost` гарсан. `outputs/validation_overlap_36_45/` дотор 271 кадрын MP4, CSV, timestamp-тай `lost` event бий; энэ клипийн timestamp-д эх видеоны 36 секундийг нэмнэ. Эх бичлэгийн 1086, 1120, 1148-р кадарт box цагаан тамирчин дээр байгааг, 1241-р кадарт баталгаагүй болсон тул box арилсныг зургаар шалгасан. Буруу хүн рүү автомат ID шилжилт гараагүй; олон кадр баталгаагүй үлдсэн нь энэ heuristic-ийн бодит хязгаар. `lost`-ийн дараа ч цагаан тамирчин давхцсан хэвээр байвал prompt тод кадр хүртэл хүлээнэ.

**Бүтэн видеоны шинэ шалгалт:** Эх MP4-ийн бүх 3954 кадрыг 300, 400, 510-р кадрын баталсан сонголтоор batch горимд боловсруулсан: 300 `unselected`, 477 `tracking`, 463 `uncertain`, 2714 `lost`. `target_id=1` сонголтоос хойш тогтвортой; түр `track_id` 7 удаа солигдсон (400, 510, 571, 615, 962, 992, 1067-р кадр). Автомат дахин холбоосын 571, 615, 962, 992, 1067-р кадрын MP4 дүрсийг үзэхэд box цагаан тамирчин дээр байсан. 1240-р кадр (41.333 секунд)-аас удаан давхцлын улмаас `lost` болж, box арилсан; энэ batch run-д дахин сонголт өгөөгүй тул цааш target-г автоматаар өөр хүн рүү шилжүүлээгүй. `outputs/validation_full_stable/` хавтаст 3954 кадрын `tracked.mp4`, мөр бүрийн `tracking.csv`, timestamp-тай `events.csv`, шалгасан кадрын `review.csv` байна. CSV-ийн tracking мөр бүр box/түр ID-тай, бусад төлөвийн координат хоосон байгааг шалгасан. Дүрсний шалгалт нь дээрх сонгосон кадруудын түүвэр бөгөөд 3954 кадр бүрийг хүнээр баталгаажуулаагүй. Бүтэн бичлэгт тасралтгүй tracking хийхийн тулд 41.33 секундээс хойш `--interactive` горимд цагаан тамирчныг дахин батлах шаардлагатай.

Өөрийн видеоны бүтэн run-д баталсан эхний гурван сонголтыг ашиглах жишээ:

```bash
python track.py --video "/mnt/c/Users/Lenovo/Videos/match.mp4" \
  --output outputs/yolov8_run01 --selections selections.match.reviewed.json \
  --device cpu --interactive
```

Батлагдаагүй удаан ID тасралт эсвэл camera cut-ийн дараа зураг гарч ирвэл цагаан тамирчны box эсвэл зурган дээрх ID-г батална. YOLOv8-ийн box хуучин YOLOv6-тай хангалттай давхцахгүйгээс баталсан сонголт татгалзагдвал `--selections`-ийг хасаж, `--reset-selection --start-frame 300` нэмээд шинэ detector-ийн зургаас цагаан тамирчныг сонгоно. `--selections`-гүй дараагийн run-д хадгалсан эхний сонголтыг автоматаар ашиглана. Аль хэдийн ажиллаж буй process-д кодын өөрчлөлт үйлчлэхгүй.

Бодит видео авсны дараа:

1. Эхлээд 30–60 секундийн тод кадр, давхцал, кадраас гарах, camera cut агуулсан хэсэг дээр ажиллуулна.
2. `tracked.mp4`-ийг original MP4-тэй зэрэгцүүлж, **tracking гэж тэмдэглэсэн хэсэгт box яг цагаан тамирчин дээр байгаа эсэхийг** нягтална.
3. `events.csv` дэх event-үүдийг шалгаж, lost/uncertain эхэлсэн болон дахин сонгосон хугацааг тэмдэглэнэ. Зөвхөн event жагсаалт нь үнэн зөв tracking-ийн баталгаа биш.
4. `review.csv` файлд `start_seconds,end_seconds,issue,notes` баганатайгаар `id_switch`, `lost`, `false_uncertain`, `missed_cut` интервалуудыг хүний ажиглалтаар бичнэ. ID switch илэрвэл тэр үеийн кадруудын box-ийг `selections.json`-д дахин баталж шинэ run хийнэ.
5. Байнгын ID switch эсвэл давхцлыг detector анзаарахгүй байвал model/appearance хамгаалалтыг сайжруулж дахин батална. Tracking баталгаажсаны дараа хөдөлгөөний шинжилгээ нэмнэ.

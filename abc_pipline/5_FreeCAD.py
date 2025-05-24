import os
import tempfile
import subprocess   # для Inkscape
import cadquery as cq
from cadquery import exporters
from svgutils.compose import Figure, SVG, Text
import shutil

from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF


# --- входные данные ----------------------------------------------------------
STEP_FILE = r"C:\Users\culic\Desktop\retrieve~\00005418\00005418_6477e8ad02554200884775b4_step_005.step"
PAGE_TPL  = r"C:/Users/culic/Downloads/ESKD_A0_L1_Landscape.svg"  # если есть

# Извлекаем номер модели из пути
model_number = os.path.basename(os.path.dirname(STEP_FILE))

# Формируем пути для SVG и PDF
OUT_SVG = f"C:/Users/culic/RAG_toy-3/abc_pipline/drawings/{model_number}.svg"
OUT_PDF = f"C:/Users/culic/RAG_toy-3/abc_pipline/drawings/{model_number}.pdf"

# размеры листа А0 (мм)
PAGE_W, PAGE_H = 1800.0, 1000.0

# сетка 2×3
NROWS, NCOLS = 2, 3

# <<< 1. увеличенные отступы сверху/снизу, чуть меньший масштаб по умолчанию
M = dict(left=50, right=50, top=60, bottom=60)
G = dict(col=90, row=130)
SCL = 0.75
# <<< конец патча

# === 0. импорт STEP и габариты ===
if not os.path.isfile(STEP_FILE):
    raise FileNotFoundError(f"STEP-файл не найден: {STEP_FILE}")
wp   = cq.importers.importStep(STEP_FILE)
bb   = wp.val().BoundingBox()
maxd = max(bb.xlen, bb.ylen, bb.zlen)

# === 1. cell_w/cell_h, масштаб ===
usable_w = PAGE_W - M['left'] - M['right'] - (NCOLS-1)*G['col']
usable_h = PAGE_H - M['top']  - M['bottom'] - (NROWS-1)*G['row']
cell_w   = usable_w/NCOLS
cell_h   = usable_h/NROWS
view_scale = min(cell_w, cell_h)/maxd * SCL



# <<< 2. авто-подгонка, если что-то всё же вылезает
grid_w   = NCOLS * cell_w + (NCOLS - 1) * G['col']
grid_h   = NROWS * cell_h + (NROWS - 1) * G['row']
# помещаем сетку строго внутри отступов M по горизонтали и вертикали
offset_x = M['left']   + (PAGE_W - M['left'] - M['right'] - grid_w) / 2
offset_y = M['bottom'] + (PAGE_H - M['top']  - M['bottom'] - grid_h) / 2


shrink = min(
    (PAGE_W - M['left'] - M['right']) / grid_w,
    (PAGE_H - M['top']  - M['bottom']) / grid_h,
    1.0
)
view_scale *= shrink
# <<< конец патча

# === 2. рендерим 6 видов в SVG-файлики ===
VIEWS = [
    ("Front",  ( 0,  0,  1)),
    ("Top",    ( 0, -1,  0)),
    ("Right",  ( 1,  0,  0)),
    ("Left",   (-1,  0,  0)),
    ("Bottom", ( 0,  1,  0)),
    ("Back",   ( 0,  0, -1)),
]
tmpdir     = tempfile.mkdtemp(prefix="cq_")
small_svgs = []

for name, dirvec in VIEWS:
    fn = os.path.join(tmpdir, f"{name}.svg")
    exporters.export(wp, fn, opt={
        "width":       int(cell_w * 0.8),
        "height":      int(cell_h * 0.8),
        "projectionDir": dirvec,
        "showHidden":  False,
        "strokeWidth": 0.2,
        "scale":       view_scale,   # если требуется передать в export
    })
    small_svgs.append((name, fn))

# === 3. компонуем лист и подписываем ===
elements = []
for idx, (name, svgpath) in enumerate(small_svgs):
    row, col = divmod(idx, NCOLS)
    x_mm = offset_x + col * (cell_w + G['col'])
    # (NROWS-1-row) чтобы row=0 был сверху, row=1 снизу
    y_mm = offset_y + (NROWS-1-row) * (cell_h + G['row'])
    elements.append(SVG(svgpath).move(x_mm, y_mm))
    txt = Text(
        f"{name} view",
        x_mm + cell_w/2,
        y_mm + cell_h + 8,
        size=20,
        font='Times-Roman',
        weight="bold",
        anchor="middle"
    )
    elements.append(txt)

# === 3.1. аннотация масштаба ===
# учитываем, что вы уменьшаете вью на 20% (т. е. VIEW_PAD = 0.8)
# ваш view_scale уже включает SCL, shrink и фактически 0.8 уменьшение по width/height,
# так что масштабный коэффициент в модели к чертежу:
scale_ratio = 1 / view_scale

# рисуем подпись "Scale 1:N" в левом нижнем углу внутри полей M
scale_txt = Text(
    f"Scale 1:{scale_ratio:.2f}",
    M['left'],                 # X = отступ слева
    M['bottom'] / 2,           # Y = половина отступа снизу
    size=24,                   # сделайте размер побольше, если нужно
    weight="bold",             # жирный шрифт
    anchor="start"             # выравнивание по левому краю
)
elements.append(scale_txt)

Figure(f"{PAGE_W}mm", f"{PAGE_H}mm", *elements).save(OUT_SVG)
print("SVG готов:", OUT_SVG)

# === 4. конвертация SVG → PDF на Python ===
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm

# === 4. конвертация SVG → PDF с правильным pagesize ===
drawing = svg2rlg(OUT_SVG)

# Создаём PDF-контекст с тем же размером, что и наш SVG-лист
c = canvas.Canvas(OUT_PDF, pagesize=(PAGE_W * mm, PAGE_H * mm))
# Рисуем весь drawing в точке (0,0)
renderPDF.draw(drawing, c, 0, 0)
c.showPage()
c.save()

print("PDF готов:", OUT_PDF)
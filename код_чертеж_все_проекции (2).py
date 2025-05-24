import sys
import os
import FreeCAD
import FreeCADGui
import Part
import TechDraw
import TechDrawGui
import Import


# must run *before* creating the TechDraw page or views
view_prefs = FreeCAD.ParamGet("User parameter:BaseApp/Preferences/Mod/TechDraw/View")
# set label font size to 5 mm:
view_prefs.SetFloat("FontSize", 5.0)
# set label font face to Arial:
view_prefs.SetString("FontName", "Arial")

# 1. Для размеров (Dimensions):
dim_prefs = FreeCAD.ParamGet("User parameter:BaseApp/Preferences/Mod/TechDraw/Dimensions")
# задаём размер шрифта в миллиметрах (например, 3 мм)
dim_prefs.SetFloat("FontSize", 3.0)  # мм :contentReference[oaicite:0]{index=0}

# 2. Для подписей (Label Font):
label_prefs = FreeCAD.ParamGet("User parameter:BaseApp/Preferences/Mod/TechDraw")
# задаём размер шрифта меток (например, 5 мм)
label_prefs.SetFloat("LabelFontSize", 5.0)  # мм :contentReference[oaicite:1]{index=1}

# === Открываем или создаём новый документ ===
doc_name = "MyDrawing"
doc = FreeCAD.getDocument(doc_name) if doc_name in FreeCAD.listDocuments() else FreeCAD.newDocument(doc_name)
FreeCAD.setActiveDocument(doc.Name)

# === Загружаем STEP ===

# заменяем путь до нашей модельки
step_file = r"C:/Users/culic/Desktop/retrieve~/00009735/00009735_bdb26e9a610c417cbccd6c20_step_012.step"
if not os.path.exists(step_file):
    raise FileNotFoundError(f"Файл {step_file} не найден!")



Import.insert(step_file, doc.Name)
doc.recompute()

if not doc.Objects:
    raise ValueError("Ошибка: Файл STEP загружен, но объекты не найдены!")

# === Берём первый объект (деталь) ===
obj = doc.Objects[0]

# === Создаём страницу TechDraw ===
page = doc.addObject('TechDraw::DrawPage', 'Page')
template = doc.addObject('TechDraw::DrawSVGTemplate', 'Template')

# вот здесь указываем свой путь:
template.Template = r"C:\Users\culic\Downloads\ESKD_A0_L1_Landscape.svg"

page.Template = template
doc.recompute()


# === Определяем размеры листа ===
try:
    page.calculatePageSize()
    page_width = float(page.PageWidth)
    page_height = float(page.PageHeight)
    if not page_width or not page_height:
        page_width, page_height = 1189.0, 841.0
except:
    page_width, page_height = 1189.0, 841.0





print(f"Размер листа: {page_width} × {page_height} мм")

# === Габариты детали ===
bbox = obj.Shape.BoundBox
width = bbox.XMax - bbox.XMin
height = bbox.YMax - bbox.YMin
depth = bbox.ZMax - bbox.ZMin
model_max_dim = max(width, height, depth)
print(f"Габариты детали: {width:.1f} × {height:.1f} × {depth:.1f} мм")

# === Определяем сетку для размещения видов ===
nrows, ncols = 2, 3

margin_left, margin_right = 50, 50
margin_top, margin_bottom = 50, 50
col_spacing, row_spacing = 90, 130  # увеличенные отступы

usable_width = page_width - margin_left - margin_right - (ncols - 1) * col_spacing
usable_height = page_height - margin_top - margin_bottom - (nrows - 1) * row_spacing

cell_w = usable_width / ncols
cell_h = usable_height / nrows
print(f"Ячейка сетки: {cell_w:.1f} × {cell_h:.1f} мм")

# === Определяем масштаб вида ===
scale_factor = 0.8
view_scale = scale_factor * (min(cell_w, cell_h) / model_max_dim)
if view_scale > 10:
    view_scale = 10

# === Центрируем виды на листе ===
grid_width = ncols * (cell_w + col_spacing) - col_spacing
grid_height = nrows * (cell_h + row_spacing) - row_spacing
center_offset_x = (page_width - grid_width) / 2 + 80
center_offset_y = (page_height - grid_height) / 2 + 80
print(f"Смещение для центрирования: X={center_offset_x}, Y={center_offset_y}")

# === Задаём виды и их направления ===
views_data = [
    ("FrontView", (0, 0, 1)),
    ("BackView", (0, 0, -1)),
    ("LeftView", (-1, 0, 0)),
    ("RightView", (1, 0, 0)),
    ("TopView", (0, -1, 0)),
    ("BottomView", (0, 1, 0)),
]

# Список для хранения созданных видов
all_views = []


annotation_coords = [
    (180.0, 432.75),
    (573.0, 432.75),
    (966.0, 432.75),
    (180.0, -2.75),
    (573.0, -2.75),
    (966.0, -2.75),
]

# === Создаём виды на странице ===
for idx, (vname, vdir) in enumerate(views_data):
    row = 1 - (idx // 3)
    col = idx % 3

    x_pos = margin_left + col * (cell_w + col_spacing) + center_offset_x
    y_pos = margin_bottom + row * (cell_h + row_spacing) + center_offset_y

    view = doc.addObject('TechDraw::DrawViewPart', vname)
    view.Source = [obj]
    view.Direction = vdir
    view.ScaleType = "Custom"
    view.Scale = view_scale
    page.addView(view)
    view.X = x_pos
    view.Y = y_pos
    all_views.append(view)

    # --- Создаём аннотацию после размещения вида ---
    annotation = doc.addObject('TechDraw::DrawViewAnnotation', f"Ann_{vname}")
    annotation.Text = [f"{vname} View"]
    annotation.X = float(view.X)
    annotation.Y = float(view.Y) - cell_h/2 - 20  # ниже центра вида на полвысоты ячейки + отступ
    page.addView(annotation)


    page.addView(view)
    view.X = x_pos
    view.Y = y_pos

    all_views.append(view)

doc.recompute()

# === Добавляем дименшены для каждого вида ===
# В данном примере для каждого вида создаются два дименшена:
# - Горизонтальный: сдвинут ниже вида (отображает протяжённость по X)
# - Вертикальный: сдвинут левее вида (отображает протяжённость по Y)
# Примечание: идентификаторы ребер ("Edge1", "Edge2", "Edge3", "Edge4") могут потребовать корректировки!

# for view in all_views:
#     #Горизонтальный размер
#     h_dim = doc.addObject("TechDraw::DrawViewDimension", f"Dim_{view.Name}_H")
#     h_dim.Type = "Distance"
#     h_dim.References2D = [(view, "Edge0"), (view, "Edge8")]
#     h_dim.X = float(view.X)        # Используем числовое значение
#     h_dim.Y = float(view.Y) - 200   # Преобразуем view.Y в число и вычитаем 10
#     page.addView(h_dim)

    # # Вертикальный размер
    # v_dim = doc.addObject("TechDraw::DrawViewDimension", f"Dim_{view.Name}_V")
    # v_dim.Type = "Distance"
    # v_dim.References2D = [(view, "Edge3"), (view, "Edge4")]
    # ten_mm = FreeCAD.Units.Quantity("10 mm")
    # v_dim.X = float(view.X) - ten_mm.Value
    # v_dim.Y = float(view.Y)
    # page.addView(v_dim)


doc.recompute()

# === Экспортируем чертёж ===
export_svg_path = r"C:\Users\culic\Downloads\exported_drawing.svg"
TechDrawGui.exportPageAsSvg(page, export_svg_path)

export_pdf_path = r"C:\Users\culic\Downloads\exported_drawing.pdf"
TechDrawGui.exportPageAsPdf(page, export_pdf_path)

print("Чертёж успешно создан и выведен в файлы:")
print("SVG:", export_svg_path)
print("PDF:", export_pdf_path)
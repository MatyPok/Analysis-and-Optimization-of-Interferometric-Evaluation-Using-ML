
from dataclasses import dataclass
import xml.etree.ElementTree as ET
from pathlib import Path


@dataclass
class BaseMask:
    name: str
    type_name: str
    center_x: float
    center_y: float
    units: int

@dataclass
class RectangleMask(BaseMask):
    width: float
    height: float

@dataclass
class CircleMask(BaseMask):
    radius: float

@dataclass
class PolygonMask(BaseMask):
    points: list[tuple[float, float]]


    

def get_text(elem, tag, default=None):
    child = elem.find(tag)
    return child.text if child is not None else default


def to_pixels(value, units, image_size):
    if units == 0:  # Pixely
        return value
    if units == 2:  # Procenta
        print(f"Converting {value} percent to {value * image_size / 100} pixels based on image size {image_size}")
        return value * image_size / 100



def get_image_size(image_path: Path) -> tuple[int, int]:

    import cv2

    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    image_size = (img.shape[1], img.shape[0])  # width, height

    return image_size


def parse_mask(mask_elem):
    type_name = get_text(mask_elem, "Type")
    name = mask_elem.attrib.get("Name")

    center_x = float(get_text(mask_elem, "CenterX", 0))
    center_y = float(get_text(mask_elem, "CenterY", 0))
    units = int(get_text(mask_elem, "Units", 0))

    base_kwargs = dict(
        name=name,
        type_name=type_name,
        center_x=center_x,
        center_y=center_y,
        units=units
    )

    # Rectangle
    if type_name.endswith("RectangleMask"):
        return RectangleMask(
            **base_kwargs,
            width=float(get_text(mask_elem, "Width")),
            height=float(get_text(mask_elem, "Height"))
        )

    # Circle
    if type_name.endswith("CircleMask"):
        return CircleMask(
            **base_kwargs,
            radius=float(get_text(mask_elem, "R"))
        )

    # Polygon
    if type_name.endswith("PolygonMask"):
        points = []
        points_elem = mask_elem.find("Points")
        if points_elem is not None:
            for p in points_elem.findall("Point"):
                points.append(
                    (float(p.attrib["X"]), float(p.attrib["Y"]))
                )

        return PolygonMask(
            **base_kwargs,
            points=points
        )

    print(f"Neznámý typ masky: {type_name}")
    return None



def create_masks_in_directory(mask_info: list, image_size: tuple, output_path: str, origin_mode: str = "top-left"):
    import cv2
    import numpy as np
    
    # Vytvoření černého obrázku
    mask_img = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
    
    # Precompute image centre
    img_w, img_h = image_size
    img_cx, img_cy = img_w / 2.0, img_h / 2.0

    for mask in mask_info:
        if isinstance(mask, RectangleMask):
            # Center je absolutní pozice
            center_x_px = to_pixels(mask.center_x, mask.units, image_size[0])
            center_y_px = to_pixels(mask.center_y, mask.units, image_size[1])

            half_width = to_pixels(mask.width, mask.units, image_size[0]) / 2
            half_height = to_pixels(mask.height, mask.units, image_size[1]) / 2
            
            left = int(center_x_px - half_width)
            top = int(center_y_px - half_height)
            right = int(center_x_px + half_width)
            bottom = int(center_y_px + half_height)
            
            cv2.rectangle(mask_img, (left, top), (right, bottom), 255, -1)
        
        elif isinstance(mask, CircleMask):
            # Center je absolutní pozice v pixelech (units=0 nebo 2)
            center_x_px = to_pixels(mask.center_x, mask.units, image_size[0])
            center_y_px = to_pixels(mask.center_y, mask.units, image_size[1])
            radius_px = to_pixels(mask.radius, mask.units, min(image_size))  # Radius je vždy v pixelech
            
            cv2.circle(mask_img, (int(center_x_px), int(center_y_px)), int(radius_px), 255, -1)
        
        elif isinstance(mask, PolygonMask):
            points_px = []
            # Pro units=0, body jsou offsety od center, posun do středu
            # Pro units!=0, body jsou absolutní, bez posunu
            if mask.units == 0:
                # Body jsou offsety od center
                xs = [p[0] for p in mask.points]
                ys = [p[1] for p in mask.points]
                bbox_center_x = (min(xs) + max(xs)) / 2
                bbox_center_y = (min(ys) + max(ys)) / 2
                mask_center_x = mask.center_x + bbox_center_x
                mask_center_y = mask.center_y + bbox_center_y
                offset_x = img_cx - mask_center_x
                offset_y = img_cy - mask_center_y
            else:
                offset_x = 0
                offset_y = 0
            
            for x, y in mask.points:
                px = to_pixels(x, mask.units, image_size[0]) + offset_x
                py = to_pixels(y, mask.units, image_size[1]) + offset_y

                points_px.append([int(px), int(py)])

            if points_px:
                pts = np.array(points_px, np.int32)
                cv2.fillPoly(mask_img, [pts], 255)
    
    # Uložení jako BMP
    cv2.imwrite(output_path, mask_img)


if __name__ == "__main__":

    mask_dir = Path("data/interferogramy_meopta/Meopta/složená/")
    
    for mask_file in mask_dir.glob("*.mask"):
        base_name = mask_file.stem  # např. 'a', 'b', 'c'
        
        # Najít odpovídající obrázky (vynech masku samotnou)
        image_files = [f for f in mask_file.parent.glob(f"{base_name}*.bmp") if f.name != f"{base_name}.bmp"] + list(mask_file.parent.glob(f"{base_name}*.tiff"))
        
        if not image_files:
            print(f"Žádné obrázky pro masku {mask_file}")
            continue
        
        # Získat velikost z prvního obrázku (ale použijeme pevnou velikost pro konzistenci)
        image_size = get_image_size(image_files[0])
        #image_size = (1184, 1184)  # Pevná velikost pro všechny masky
        
        # Parsovat masku
        tree = ET.parse(mask_file)
        root = tree.getroot()
        masks = [parse_mask(m) for m in root.findall("mask")]
        masks = [m for m in masks if m is not None]
        
        # Vybrat režim původu souřadnic:
        # - body jsou absolutní pixely
        origin_mode = "top-left"

        # Vytvořit a uložit masku
        output_path = mask_file.parent / f"{base_name}.bmp"
        create_masks_in_directory(masks, image_size, str(output_path), origin_mode=origin_mode)

        print(f"Maska pro {base_name} uložena jako {output_path} (origin_mode={origin_mode})")
